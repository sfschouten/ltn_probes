import os
import math
from itertools import chain
from collections import namedtuple

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


class LogitsToPredicate(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.sigmoid(logits)
        out = torch.sum(probs * l, dim=1)
        return out


def create_axioms(x, labels, frames, frame_roles, frames_roles_count, predicates, device):
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
        f'{name}_{v}': ForAll(
            ltn.diag(x_var, l_var), op(p_var(x_var, l_var)),
            cond_vars=[l_var], cond_fn=lambda t: t.value == v
        )
        for name, (p_var, x_var, l_var) in var_pairs.items()
        for v, op in [(True, Id), (False, Not)]
    }

    # presence of frame element implies presence of frame
    one = ltn.Variable('one', torch.ones((1,), device=device))
    for i, j in frames_roles_count.nonzero().tolist():
        frame_role_name = frame_roles[i]
        frame_name = frames[j]
        p_frame_role, x_frame_role, l_frame_role = var_pairs[frame_role_name]
        p_frame, x_frame, l_frame = var_pairs[frame_name]
        closed_formulas[f'{frame_role_name}->{frame_name}'] = ForAll(
            [one] + ltn.diag(x_frame_role, x_frame),
            Implies(p_frame_role(x_frame_role, one), p_frame(x_frame, one))
        )

    # TODO add 'presence of frame implies presence of _core_ frame elements' formulas

    # TODO add formulas that capture frame relations

    return closed_formulas


def train_ltn(dataloader_train, dataloader_test, frames, frame_roles, frames_roles_count, args, ndim):
    if args.log_neptune:
        import neptune
        run = neptune.init_run(
            project=os.getenv('NEPTUNE_PROJECT'),
            api_token=os.getenv('NEPTUNE_API_KEY'),
        )
        params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
                  "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device}
        run["parameters"] = params

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
        name: ltn.Predicate(LogitsToPredicate(nn.Linear(args.probe_attn_n_value_d, 1))).to(args.probe_device)
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

            closed_formulas = create_axioms(
                x, labels, frames, frame_roles, frames_roles_count, predicates, args.probe_device)

            sat_agg = SatAgg(*list(closed_formulas.values()))
            loss = 1 - sat_agg

            # descend gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log_neptune:
                for key, value in closed_formulas.items():
                    run[f'train/{key}'].append(value.value)
                run["train/loss"].append(loss.cpu().detach())

    def test_epoch():
        for p in predicates.values():
            p.eval()
        attn.eval()

        with torch.no_grad():
            for hs, sentences, labels in dataloader_test:
                hs = hs.to(args.probe_device)

                x = forward(hs)

                closed_formulas = create_axioms(
                    x, labels, frames, frame_roles, frames_roles_count, predicates, args.probe_device)

                sat_agg = SatAgg(*list(closed_formulas.values()))
                loss = 1 - sat_agg

                if args.log_neptune:
                    for key, value in closed_formulas.items():
                        run[f'test/{key}'].append(value.value)
                    run["test/loss"].append(loss)
        return sat_agg

    # test once to see random performance
    test_epoch()
    for e in tqdm(range(args.nr_epochs)):
        train_epoch()
        test_sat = test_epoch()
        print(f'epoch {e}, test_sat={test_sat}')

    if args.log_neptune:
        run.stop()


def main(args, generation_args):
    print(f'CUDA available? {torch.cuda.is_available()}')
    print(f'Nr. of CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Name of first device: {torch.cuda.get_device_name(0)}')
    print()

    data, frames, frame_roles, frames_roles_count = get_framenet_dataset(None)
    dataset = data.train_test_split(test_size=.2, shuffle=False)

    def trim_hidden_states(hs):
        mask = np.isfinite(hs)  # (nr_samples, nr_layers, nr_tokens, nr_dims)
        mask = mask.all(axis=3)
        token_cnt = mask.sum(axis=2)
        trim_i = token_cnt.max()
        print(f'Trimming to {trim_i} from {hs.shape[2]}.')
        return hs[:, :, :trim_i, :]

    # load dataset and hidden states
    hs = trim_hidden_states(load_single_generation(vars(generation_args)))

    # normalize
    # hs[~np.isfinite(hs)] = np.nan
    # hs -= np.nanmean(hs, axis=0, keepdims=True)
    # hs /= np.nanstd(hs, axis=0, keepdims=True)

    ndim = hs.squeeze().shape[-1]

    train_data = dataset['train']
    test_data = dataset['test']

    if args.random_baseline:
        hs = np.random.randn(len(dataset), 32, ndim)
    elif args.shuffled_baseline:
        np.random.shuffle(hs)

    # train LTN probe
    hs_t = torch.Tensor(hs).squeeze()
    hs_dataset_train = CustomDataset(hs_t[:len(train_data)], train_data)
    hs_dataset_test = CustomDataset(hs_t[len(train_data):], test_data)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_t)
    hs_dataloader_train = DataLoader(hs_dataset_train, batch_size=batch_size, shuffle=True)
    hs_dataloader_test = DataLoader(hs_dataset_test, batch_size=len(test_data), shuffle=True)

    train_ltn(hs_dataloader_train, hs_dataloader_test, frames, frame_roles, frames_roles_count, args, ndim)


if __name__ == '__main__':
    parser = get_parser()
    _generation_args, _ = parser.parse_known_args()
    parser.add_argument("--nr_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_weight_decay", type=float, default=0.05)
    parser.add_argument("--probe_attn_n_key_query_d", type=int, default=128)
    parser.add_argument("--probe_attn_n_value_d", type=int, default=128)
    parser.add_argument("--probe_attn_bias", type=bool, default=True)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    parser.add_argument("--shuffled_baseline", action='store_true')
    _args = parser.parse_args()
    main(_args, _generation_args)

