from tqdm import tqdm

import ltn
from utils import get_parser, load_single_generation, get_dataset, get_dataloader
from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import neptune


class SPO_Attention(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)

        self.n_embd = n_embd

        one = torch.ones(1, 1, n_embd)

        s = torch.randn_like(one)  # TODO these might need a random initialization
        p = torch.randn_like(one)
        o = torch.randn_like(one)
        self.q = nn.Parameter(torch.cat((s, p, o), dim=1))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)
        x = x.to("cuda")
        k, v = self.c_attn(x).split(self.n_embd, dim=2)

        mask = mask.all(dim=-1).unsqueeze(1).expand(-1, 3, -1)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(self.q, k, v, attn_mask=mask)
        else:
            # manual implementation of attention
            att = (self.q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, T, T) x (B, T, hs) -> (B, T, hs)
        return y.squeeze()


# we define predicate P
class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes=(4096, 3)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])  #

    def forward(self, x, training=False):
        """
        Method which defines the forward phase of the neural network for our multi class classification task.
        In particular, it returns the logits for the classes given an input example.

        :param x: the features of the example
        :param training: whether the network is in training mode (dropout applied) or validation mode (dropout not applied)
        :return: logits for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
        logits = self.linear_layers[-1](x)
        return logits


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """

    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x, l, training=False):
        logits = self.logits_model(x, training=training)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out


def train_ltn(dataloader, dataloader_test, args, ndim):
    run = neptune.init_run(
        project="frankissimo/ltnprobing",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTY1M2U5Ni05ZTU0LTQ0YjAtYWM0OC1jNzUyZTIwOWNiNDQifQ==",
    )  # your credentials

    params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
              "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device}
    run["parameters"] = params

    attn = SPO_Attention(ndim)
    attn = attn.to(args.probe_device)

    mlp = MLP()
    Model = ltn.Predicate(LogitsToPredicate(mlp))
    # Action = ltn.Predicate(LogitsToPredicate(mlp))
    # Object = ltn.Predicate(LogitsToPredicate(mlp))
    Model = Model.to(args.probe_device)
    # Action = Action.to("cuda")
    # Object = Object.to("cuda")

    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()
    parameters = []
    parameters.extend([f for f in Model.parameters()])
    # parameters.extend([f for f in Action.parameters()])
    # parameters.extend([f for f in Object.parameters()])
    parameters.extend([f for f in attn.parameters()])
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    for epoch in range(args.nr_epochs):

        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            hidden_states, = batch

            hidden_states = hidden_states.to("cuda")
            spo = attn(hidden_states)
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])
            Subject_l = ltn.Constant(torch.tensor([1, 0, 0]))
            Object_l = ltn.Constant(torch.tensor([0, 1, 0]))
            Action_l = ltn.Constant(torch.tensor([0, 0, 1]))
            sat_agg = SatAgg(
                Forall(x, Model(x, Subject_l)),
                # Forall(x, Not(Action(x))),
                # Forall(x, Not(Object(x))),

                Forall(y, Model(y, Action_l)),
                # Forall(y, Not(Subject(y))),
                # Forall(y, Not(Object(y))),

                Forall(z, Model(z, Object_l))
                # Forall(z, Not(Subject(z))),
                # Forall(z, Not(Action(z)))
                # Forall(ltn.diag(x,y,z),And(And(Subject(x), Action(y)), Object(z)))
            )

            loss = 1. - sat_agg
            loss.backward()
            optimizer.step()
            run["train/subject_satisability"].append(Forall(x, Model(x, Subject_l)).value)
            run["train/Action_satisability"].append(Forall(y, Model(y, Action_l)).value)
            run["train/Object_satisability"].append(Forall(z, Model(z, Object_l)).value)
            run["train/loss"].append(1. - sat_agg)

    Model.eval()
    attn.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            hidden_states, = batch

            hidden_states = hidden_states.to("cuda")
            spo = attn(hidden_states)
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])
            Subject_l = ltn.Constant(torch.tensor([1, 0, 0]))
            Object_l = ltn.Constant(torch.tensor([0, 1, 0]))
            Action_l = ltn.Constant(torch.tensor([0, 0, 1]))
            sat_agg = SatAgg(
                Forall(x, Model(x, Subject_l)),
                # Forall(x, Not(Action(x))),
                # Forall(x, Not(Object(x))),

                Forall(y, Model(y, Action_l)),
                # Forall(y, Not(Subject(y))),
                # Forall(y, Not(Object(y))),

                Forall(z, Model(z, Object_l))
                # Forall(z, Not(Subject(z))),
                # Forall(z, Not(Action(z)))
                # Forall(ltn.diag(x,y,z),And(And(Subject(x), Action(y)), Object(z)))
            )

            loss = 1. - sat_agg

            run["test/subject_satisability"].append(Forall(x, Model(x, Subject_l)).value)
            run["test/Action_satisability"].append(Forall(y, Model(y, Action_l)).value)
            run["test/Object_satisability"].append(Forall(z, Model(z, Object_l)).value)
            run["test/loss"].append(1. - sat_agg)

            # print(loss)

    run.stop()


def main(args, generation_args):
    # load dataset and hidden states 
    hs_train = load_single_generation(generation_args)
    hs_test = load_single_generation(generation_args)

    tokenizer = AutoTokenizer.from_pretrained(generation_args.model_name, use_fast=True)
    dataset_train, tokenized_train, dataset_test, tokenized_test = get_dataset(tokenizer)

    for hs_row, sample in zip(hs_train, dataset_train):
        # print(hs_row)
        # print(sample)

        # sample_enc in case we need to know which token is part of what word.
        sample_enc = tokenizer(sample['sentence'])

    for hs_row, sample in zip(hs_test, dataset_train):
        # print(hs_row)
        # print(sample)

        # sample_enc in case we need to know which token is part of what word.
        sample_enc = tokenizer(sample['sentence'])

    # train LTN probe
    hs_train_t = torch.Tensor(hs_train).squeeze()
    batch, nr_tokens, ndim = hs_train_t.shape
    hs_dataset_train = TensorDataset(hs_train_t)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_train_t)
    hs_dataloader_train = DataLoader(hs_dataset_train, batch_size=batch_size)

    # test
    hs_test_t = torch.Tensor(hs_test).squeeze()
    batch, nr_tokens, ndim = hs_test_t.shape
    hs_dataset_test = TensorDataset(hs_test_t)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_test_t)
    hs_dataloader_test = DataLoader(hs_dataset_test, batch_size=batch_size)
    train_ltn(hs_dataloader_test, hs_dataloader_test, args, ndim)


if __name__ == "__main__":
    parser = get_parser()
    generation_args, _ = parser.parse_known_args()
    # We'll also add some additional args for evaluation
    parser.add_argument("--nr_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--probe_device", type=str, default='cuda')
    args = parser.parse_args()
    main(args, generation_args)
