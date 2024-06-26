import os
import math
from itertools import chain
from collections import namedtuple
import csv

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ltn
from ltn.fuzzy_ops import UnaryConnectiveOperator

from customdataloader import CustomDataset
from utils import get_framenet_dataset, get_parser, load_single_generation

from dotenv import load_dotenv
load_dotenv()


class Identity(UnaryConnectiveOperator):
    """ Dummy operator that doesn't do anything. """
    def __call__(self, x):
        return x


VarPair = namedtuple('VarPair', ['predicate', 'data', 'label'])

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Id = ltn.Connective(Identity())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
ForAll = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
# SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=4.0))
SatAgg = ltn.fuzzy_ops.SatAgg()


class FrameRoleAttention(nn.Module):
    def __init__(self, nr_frame_roles, n_emb_d, n_key_query_d=None, n_value_d=None, bias=True):
        super().__init__()

        # key, value projections
        self.v_proj = nn.Linear(n_emb_d, n_value_d, bias=bias)
        self.k_proj = nn.Linear(n_emb_d, n_key_query_d, bias=bias)
        # frame role vectors (queries)
        self.frame_roles = nn.Embedding(nr_frame_roles, n_key_query_d)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, return_attn_values=False):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)

        q = self.frame_roles.weight     # nr_frame_roles, n_embd
        k = self.k_proj(x)              # batch, tokens, n_embd
        v = self.v_proj(x)              # batch, tokens, n_embd

        # attend
        mask = mask.all(dim=-1).unsqueeze(1).expand(-1, 1, -1)
        if self.flash and not return_attn_values:
            # efficient attention using Flash Attention CUDA kernels
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask).squeeze()
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~mask, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = (att @ v).squeeze()     # batch, nr_frame_roles, n_embd
            return y, att if return_attn_values else y


def create_axioms(x, labels, frames, frame_roles, frames_roles_count, frames_implications, predicates):
    # create tuples of predicates (frames and frame roles), their inputs, and labels
    var_pairs = {}
    for i, name in enumerate(frames + frame_roles):
        var_pairs[name] = VarPair(
            predicates[name],
            ltn.Variable(f'x_{name}', x[:, i]),
            ltn.Variable(f'l_{name}', labels[:, i]),
        )

    # multi-label supervision
    # for all pairs of (x, l) in our batch, if l==1 then P(x) if l==0 then -P(x)
    closed_formulas = {
        ('sv', f'{name}_{v}'): ForAll(
            ltn.diag(x_var, l_var), op(p_var(x_var)),
            cond_vars=[l_var], cond_fn=lambda t: t.value == v
        )
        for name, (p_var, x_var, l_var) in var_pairs.items()
        for v, op in [(True, Id), (False, Not)]
    }

    # presence of frame element implies presence of frame
    closed_formulas |= {
        ('fe-f', f'{frame_roles[i]}->{frames[j]}'): (
            lambda p_frame_role, x_frame_role, _1, p_frame, x_frame, _2: ForAll(
                ltn.diag(x_frame_role, x_frame),
                Implies(p_frame_role(x_frame_role), p_frame(x_frame))
            )
        )(*var_pairs[frame_roles[i]], *var_pairs[frames[j]])
        for i, j in frames_roles_count.nonzero().tolist()
    }

    # formulas that capture frame relations
    closed_formulas |= {
        ('f-f', f'{frames[i]}->{frames[j]}'): (
            lambda p_frame1, x_frame1, _1, p_frame2, x_frame2, _2: ForAll(
                ltn.diag(x_frame1, x_frame2),
                Implies(p_frame1(x_frame1), p_frame2(x_frame2))
            )
        )(*var_pairs[frames[i]], *var_pairs[frames[j]])
        for i, j in frames_implications.nonzero().tolist()
    }

    # TODO add "presence of frame implies presence of _core_ frame elements" formulas?

    return var_pairs, closed_formulas


def calc_metrics(var_pairs, closed_formulas, to_calc, dummy_baseline=False):
    metrics = {}
    for name, (p_var, x_var, l_var) in var_pairs.items():
        if name not in to_calc:
            continue

        labels = l_var.value.squeeze()
        if dummy_baseline:
            pred = torch.ones_like(labels.float())
            assert False in labels.mode()[0], "False must be the most common class"
        else:
            pred = p_var.model(x_var.value).squeeze()

        correct = (pred - labels.float()).abs() < 0.5
        tp = torch.sum(correct[labels]).item()
        fp = torch.sum(~correct[labels]).item()
        tn = torch.sum(correct[~labels]).item()
        fn = torch.sum(~correct[~labels]).item()

        metrics[('var', '', name, 'tp')] = tp / len(labels)
        metrics[('var', '', name, 'fp')] = fp / len(labels)
        metrics[('var', '', name, 'tn')] = tn / len(labels)
        metrics[('var', '', name, 'fn')] = fn / len(labels)
        metrics[('var', '', name, 'pr')] = tp / (tp + fp) if tp + fp > 0 else -1
        metrics[('var', '', name, 're')] = tp / (tp + fn) if tp + fn > 0 else -1
        metrics[('var', '', name, 'f1')] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else -1

    for cf_key, cf_val in closed_formulas.items():
        a, b = cf_key
        metrics[('cf', a, b, 'sat')] = cf_val.value.detach().cpu().numpy()

    return metrics


def train_ltn(dataloader_train, dataloader_valid, dataloader_test,
              frames, frame_roles, frames_roles_count, frames_implications, args, generation_args, ndim):
    if args.log_neptune:
        import neptune
        run = neptune.init_run(
            project=os.getenv('NEPTUNE_PROJECT'),
            api_token=os.getenv('NEPTUNE_API_KEY'),
        )
        run["probing_args"] = vars(args)
        run["generation_args"] = vars(generation_args)

    # attention mechanism
    attn = FrameRoleAttention(
        nr_frame_roles=len(frame_roles), n_emb_d=ndim, n_key_query_d=args.probe_attn_n_key_query_d,
        n_value_d=args.probe_attn_n_value_d, bias=args.probe_attn_bias
    ).to(args.probe_device)

    # we have count of frame frame_role co-occurrences
    # construct a matrix which creates a representation for a frame by averaging the representations for frame roles
    frames_roles_mat = frames_roles_count.sign().float().to(args.probe_device)       # nr_frame_roles, nr_frames
    frames_roles_mat /= frames_roles_mat.sum(dim=0, keepdims=True)
    frames_roles_mat = torch.nan_to_num(frames_roles_mat)

    # predicates, one for each frame and frame-role
    predicates = {
        name: ltn.Predicate(nn.Sequential(
            nn.Linear(args.probe_attn_n_value_d, 1),
            nn.Sigmoid()
        )).to(args.probe_device)
        for name in frames + frame_roles
    }

    # create optimizer
    optimizer = torch.optim.Adam(
        chain(attn.parameters(), *[p.parameters() for p in predicates.values()]),
        lr=args.lr, weight_decay=args.probe_weight_decay
    )

    def forward(hs):
        attn_out = attn(hs, return_attn_values=False)               # batch, nr_frame_roles, ndim
        x_frames = (attn_out.mT @ frames_roles_mat).mT              # batch, ndim, nr_frames
        return torch.cat(tensors=(x_frames, attn_out), dim=1)       # batch, nr_frames+nr_roles, ndim

    def train_epoch():
        for p in predicates.values():
            p.train()
        attn.train()

        for hs, _, labels in dataloader_train:
            hs = hs.to(args.probe_device)

            x = forward(hs)

            _, closed_formulas = create_axioms(
                x, labels, frames, frame_roles, frames_roles_count, frames_implications, predicates)

            sat_agg = SatAgg(*list(closed_formulas.values()))
            loss = 1 - sat_agg

            # descend gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log_neptune:
                run["train/loss"].append(loss.cpu().detach())

    FIELDNAMES = ['group', 'prefix', 'epoch', 'type1', 'type2', 'name', 'metric', 'value']

    def eval_epoch(eval_gr, epoch, dataloader, prefix="", dummy_baseline=False):
        for p in predicates.values():
            p.eval()
        attn.eval()

        with torch.no_grad():
            for hs, sentences, labels in dataloader:
                hs = hs.to(args.probe_device)
                x = forward(hs)
                var_pairs, closed_formulas = create_axioms(
                    x, labels, frames, frame_roles, frames_roles_count, frames_implications, predicates)

                sat_agg = SatAgg(*list(closed_formulas.values()))
                loss = 1 - sat_agg
                metrics = calc_metrics(var_pairs, closed_formulas, frames, dummy_baseline=dummy_baseline)
                # metrics = {f'{prefix}{key}': v for key, v in metrics.items()}

        if args.log_neptune:
            for key, value in metrics.items():
                run[f'{eval_gr}/' + '_'.join((prefix,) + key)].append(value)
            run[f"{eval_gr}/loss"].append(loss)
        if args.write_metrics:
            with open(args.metrics_file, 'a') as metrics_file:
                w = csv.DictWriter(metrics_file, fieldnames=FIELDNAMES)
                for key, value in metrics.items():
                    typ1, typ2, name, metric = key
                    w.writerow({'group': eval_gr, 'prefix': prefix, 'epoch': epoch,
                                'type1': typ1, 'type2': typ2, 'name': name, 'metric': metric, 'value': value})

        return sat_agg

    if args.write_metrics:
        with open(args.metrics_file, 'w') as metrics_file:
            w = csv.DictWriter(metrics_file, fieldnames=FIELDNAMES)
            w.writeheader()

    # test once to see random performance
    eval_epoch('test', -1, dataloader_test, dummy_baseline=False, prefix='init')
    eval_epoch('test', -1, dataloader_test, dummy_baseline=True, prefix='dummy')
    for e in tqdm(range(args.nr_epochs)):
        train_epoch()
        valid_sat = eval_epoch('valid', e, dataloader_valid)
        print(f'epoch {e}, valid_sat={valid_sat}')
    test_sat = eval_epoch('test', -2, dataloader_test)
    print(f'test_sat={test_sat}')

    if args.log_neptune:
        run.stop()


def main(args, generation_args):
    print(f'CUDA available? {torch.cuda.is_available()}')
    print(f'Nr. of CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Name of first device: {torch.cuda.get_device_name(0)}')
    print()

    data, frames, frame_roles, frames_roles_count, frames_implications = get_framenet_dataset(None)
    train_eval = data.train_test_split(test_size=0.3, shuffle=False)
    valid_test = train_eval['test'].train_test_split(test_size=0.5, shuffle=False)
    train_data, train_b = train_eval['train'], 0
    valid_data, valid_b = valid_test['train'], len(train_data)
    test_data, test_b = valid_test['test'], valid_b + len(valid_data)

    def trim_hidden_states(hs):
        mask = np.isfinite(hs)  # (nr_samples, nr_tokens, nr_dims, nr_layers)
        mask = mask.all(axis=2)
        token_cnt = mask.sum(axis=1)
        trim_i = token_cnt.max()
        print(f'Trimming to {trim_i} from {hs.shape[1]}.')
        return hs[:, :trim_i, :, :]

    # load dataset and hidden states
    hs = trim_hidden_states(load_single_generation(vars(generation_args))).squeeze()
    _, ntokens, ndim = hs.shape

    if args.random_baseline:
        hs[:len(train_data)] = np.random.randn(len(train_data), ntokens, ndim)
    elif args.shuffled_baseline:
        np.random.shuffle(hs[:len(train_data)])

    # train LTN probe
    hs_t = torch.Tensor(hs)
    hs_dataset_train = CustomDataset(hs_t[:valid_b], train_data)
    hs_dataset_valid = CustomDataset(hs_t[valid_b:test_b], valid_data)
    hs_dataset_test = CustomDataset(hs_t[test_b:], test_data)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_t)
    hs_dl_train = DataLoader(hs_dataset_train, batch_size=batch_size, shuffle=True)
    hs_dl_valid = DataLoader(hs_dataset_valid, batch_size=len(valid_data), shuffle=True)
    hs_dl_test = DataLoader(hs_dataset_test, batch_size=len(test_data), shuffle=True)

    train_ltn(hs_dl_train, hs_dl_valid, hs_dl_test,
              frames, frame_roles, frames_roles_count, frames_implications, args, generation_args, ndim)


if __name__ == '__main__':
    parser = get_parser()
    _generation_args, _ = parser.parse_known_args()
    parser.add_argument("--nr_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_weight_decay", type=float, default=0)
    parser.add_argument("--probe_attn_n_key_query_d", type=int, default=128)
    parser.add_argument("--probe_attn_n_value_d", type=int, default=128)
    parser.add_argument("--probe_attn_bias", type=bool, default=True)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--write_metrics", action='store_true')
    parser.add_argument("--metrics_file", type=str, default='metrics.csv')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    parser.add_argument("--shuffled_baseline", action='store_true')
    _args = parser.parse_args()
    main(_args, _generation_args)

