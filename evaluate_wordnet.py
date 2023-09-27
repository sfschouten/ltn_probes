import csv
import math
import os
import pickle
import random

import neptune
from torch import optim
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ltn

from utils import get_parser, load_single_generation, get_dataset
from customdataloader import CustomDataset, CustomDatasetTest

from dotenv import load_dotenv

load_dotenv()


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def trim_hidden_states(hs):
    mask = np.isfinite(hs)  # (nr_samples, nr_layers, nr_tokens, nr_dims)
    mask = mask.all(axis=3)
    token_cnt = mask.sum(axis=2)
    trim_i = token_cnt.max()
    print(f'Trimming to {trim_i} from {hs.shape[2]}.')
    return hs[:, :, :trim_i, :]


class SPOAttention(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.n_embd = n_embd

        s = torch.randn(1, 1, n_embd)
        p = torch.randn(1, 1, n_embd)
        o = torch.randn(1, 1, n_embd)
        self.q = nn.Parameter(torch.cat((s, p, o), dim=1))

        self.dropout = nn.Dropout(0.5)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # self.MatrixVector = nn.Parameter(2e-4 * torch.rand([4096, 1013 + 2]), requires_grad=True)
        # self.MatrixVectorFirst = nn.Parameter(2e-4 * torch.rand([4096, 12]), requires_grad=True)
        # self.MatrixVectorSecond = nn.Parameter(2e-4 * torch.rand([4096, 43]), requires_grad=True)
        # self.MatrixVectorThird = nn.Parameter(2e-4 * torch.rand([4096, 156]), requires_grad=True)
        # self.MatrixContinent = nn.Parameter(2e-4 * torch.rand([4096, 9]), requires_grad=False)
        # self.MatrixHabitat = nn.Parameter(2e-4 * torch.rand([4096, 28]), requires_grad=True)

    def forward(self, x, return_attn_values=True):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)

        k = self.k_proj(x)

        v = self.v_proj(x)

        mask = mask.all(dim=-1).unsqueeze(1).expand(-1, 1, -1)
        if self.flash and not return_attn_values:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(self.q, k, v, attn_mask=mask)
            return y.squeeze()
        else:
            # manual implementation of attention
            att = (self.q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~mask, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = att @ v  # (B, T, T) x (B, T, hs) -> (B, T, hs)
            return y.squeeze(), att


# Funzione di attenzione basata sul prodotto scalato

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Applichiamo la maschera per evitare l'attenzione a padding

    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, value)
    return output, attention_weights


# Modello con funzione di attenzione basata sul prodotto scalato
class ScaledDotProductAttentionModel(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(ScaledDotProductAttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "La dimensione dell'embedding deve essere divisibile per il numero di teste."
        self.head_dim = embedding_dim // num_heads
        self.fc_q = nn.Linear(embedding_dim, embedding_dim)
        self.fc_k = nn.Linear(embedding_dim, embedding_dim)
        self.fc_v = nn.Linear(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, src):
        q = self.fc_q(src)
        k = self.fc_k(src)
        v = self.fc_v(src)

        # Reshape degli embedding in testa (chunks)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Calcoliamo l'attenzione per tutte le teste simultaneamente
        output, attention_weights = scaled_dot_product_attention(q, k, v)

        # Concateniamo lungo la dimensione delle teste e proiettiamo l'output attraverso un layer lineare
        output = output.view(-1, self.embedding_dim)
        output = self.fc(output)

        # Reshape dell'output per ottenere la forma 3x4096
        output = output.view(-1, self.num_heads, self.head_dim)

        return output


class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes=(4096, 4)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

        layers = [torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                  for i in range(1, len(layer_sizes))]

        # [torch.nn.init.xavier_uniform(f.weight) for f in layers]
        """
        for i in layers:
            i.weight.data = (2e-4 * torch.rand([layer_sizes[1], layer_sizes[0]]))

        """

        self.linear_layers = torch.nn.ModuleList(layers)  #

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

    def forward(self, x, l=None, training=False):
        logits = self.logits_model(x, training=training)

        """
        if l != None:
            probs = self.softmax(logits)
            out = torch.sum(probs * l, dim=1)
        else:
        """
        probs = self.softmax(logits)
        if l != None:
            out = torch.gather(probs, 1, l)
        else:
            out = probs
        return out


def create_axioms(Model, Model_continent, Model_category, Model_habitat, Model_person, Model_object, Subject_l,
                  Action_l, Object_l, score_category,
                  score_continent,
                  score_habitat, labels, x, y,
                  z, label_category, label_continent, label_habitat, label_sentence, label_sentence_macro, epoch,

                  subject, action, object, sentences):
    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall_person = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6.0), quantifier="f")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
    Forall_object = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=6.0))
    # agg_op=ltn.fuzzy_ops.AggregLogSum(weights=[10,10,1,1,1,1,1,1,1]
    #

    label_object = ltn.Variable("object",
                                torch.tensor(torch.zeros(label_sentence.value.shape[0], label_sentence.value.shape[1]),
                                             dtype=torch.int64))
    label_subject = ltn.Variable("subject",
                                 torch.tensor(torch.ones(label_sentence.value.shape[0], label_sentence.value.shape[1]),
                                              dtype=torch.int64))

    subject_positive = Forall(ltn.diag(label_sentence, x, Subject_l, label_sentence_macro),
                              Model(x, Subject_l),
                              cond_vars=[label_sentence, label_sentence_macro],
                              cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    action_positive = Forall(ltn.diag(y, Action_l), Model(y, Action_l))

    object_positive = Forall_object(ltn.diag(label_sentence, z, Object_l, label_sentence_macro),
                                    Model(z, Object_l),
                                    cond_vars=[label_sentence, label_sentence_macro],
                                    cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    subject = ltn.Variable("subject", subject)
    action = ltn.Variable("action", action)
    object = ltn.Variable("object", object)

    axiom_habitat = Forall(ltn.diag(label_sentence, score_habitat, label_habitat, label_sentence_macro),
                           Model_habitat(score_habitat, label_habitat),
                           cond_vars=[label_sentence, label_sentence_macro],
                           cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    axiom_category = Forall(ltn.diag(label_sentence, score_category, label_category, label_sentence_macro),
                            Model_category(score_category, label_category),
                            cond_vars=[label_sentence],
                            cond_fn=lambda t: t.value == 1)

    is_not_person_subject = Forall(ltn.diag(label_sentence, subject, label_object, label_sentence_macro),
                                   Model_person(subject, label_object),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 0)

    is_not_person_object = Forall(ltn.diag(label_sentence, object, label_object, label_sentence_macro),
                                  Model_object(object, label_object),
                                  cond_vars=[label_sentence],
                                  cond_fn=lambda t: t.value == 0)

    is_person_subject = Forall(ltn.diag(label_sentence, subject, label_subject, label_sentence_macro),
                               Model_person(subject, label_subject),
                               cond_vars=[label_sentence],
                               cond_fn=lambda t: t.value == 1)

    is_person_object = Forall(ltn.diag(label_sentence, object, label_subject, label_sentence_macro),
                              Model_object(object, label_subject),
                              cond_vars=[label_sentence],
                              cond_fn=lambda t: t.value == 1)

    # , subject_positive, action_positive,
    sat_agg = SatAgg(object_positive, subject_positive, action_positive, axiom_habitat,
                     axiom_category, is_not_person_subject, is_not_person_object, is_person_object, is_person_subject)

    return {
        'subject_positive': subject_positive,
        # 'SatAgg': SatAgg.agg_op.p,
        'Forall': Forall.agg_op.p,
        'Forallobject': Forall_object.agg_op.p,
        # 'subject_negative': subject_negative,
        'is_person_subject': is_person_subject,
        'is_person_object': is_person_object,
        'is_not_person_object': is_not_person_object,
        'is_not_person_object': is_not_person_object,
        # 'is_implication':is_implication,
        'axiom_category': axiom_category,
        # 'is_implication_nation':is_implication_nation,
        # 'subject_negative': subject_negative,
        'action_positive': action_positive,
        # 'action_negative': action_negative,
        'object_positive': object_positive,
        # 'object_negative': object_negative,
        'axiom_habitat': axiom_habitat,
        # 'axiom_continent': axiom_continent,
        # 'object_negative': object_negative,
        # 'all_sentence_positive': all_sentence_positive,
        # 'all_sentence_negative': all_sentence_negative,
        # 'axiom_category': axiom_category,
        # 'all_sentence_positive_implication': all_sentence_positive_implication,
        # 'all_sentence_negative_implication': all_sentence_negative_implication,
    }, sat_agg


def create_axioms_test(Model, Model_continent, Model_category, Model_habitat, Model_person, Model_object,
                       Subject_l,
                       Action_l,
                       Object_l,
                       score_category,
                       score_continent,
                       score_habitat, labels, x, y,
                       z, label_category, label_continent, label_habitat, label_sentence,
                       label_sentence_macro, epoch,
                       subject, action,
                       object,

                       Subject_l_odd,
                       Action_l_odd,
                       Object_l_odd,
                       score_category_odd,
                       score_continent_odd, score_habitat_odd,
                       labels_odd,
                       x_odd, y_odd,
                       z_odd,
                       label_category_odd,
                       label_continent_odd,
                       label_habitat_odd,
                       label_sentence_odd,
                       label_sentence_macro_odd,
                       subject_odd, action_odd, object_odd
                       ):
    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    And = ltn.Connective(ltn.fuzzy_ops.AndMin())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall_person = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6.0), quantifier="f")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
    Forall_object = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=4.0))

    subject = ltn.Variable("subject", subject)
    action = ltn.Variable("action", action)
    object = ltn.Variable("object", object)

    label_object = ltn.Variable("object",
                                torch.tensor(torch.zeros(label_sentence.value.shape[0],
                                                         label_sentence.value.shape[1]),
                                             dtype=torch.int64))
    label_subject = ltn.Variable("subject",
                                 torch.tensor(torch.ones(label_sentence.value.shape[0],
                                                         label_sentence.value.shape[1]),
                                              dtype=torch.int64))

    first_axiom_positive = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 label_sentence_macro),
        And(And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                Model(y, Action_l)),
            And(Model(z, Object_l), Model_object(object, label_subject))),
        cond_vars=[label_sentence, label_sentence_macro],
        cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    second_axiom_positive = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 score_category, label_category, label_sentence_macro),
        Implies(And(And(And(Model(x, Subject_l),
                            Model_person(subject, label_subject)), Model(y, Action_l)),
                    And(Model(z, Object_l), Model_object(object, label_subject))),
                And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                    And(Model_category(score_category, label_category),
                        Model_object(object, label_subject)))),
        cond_vars=[label_sentence, label_sentence_macro],

        cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    thirth_axiom_positive = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 score_habitat, label_habitat, label_sentence_macro),
        Implies(
            And(And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                    Model(y, Action_l)),
                And(Model(z, Object_l), Model_object(object, label_subject))),
            And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                And(Model_habitat(score_habitat, label_habitat),
                    Model_object(object, label_subject)))),
        cond_vars=[label_sentence, label_sentence_macro],
        cond_fn=lambda t, t1: torch.logical_and(t.value == 1, t1.value == 0))

    # if gordon is a carnivore then gordon is gs
    fourth_axiom_positive = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 score_habitat, label_habitat, label_sentence_macro,

                 x_odd, Subject_l_odd, y_odd, Action_l_odd, z_odd, Object_l_odd,
                 label_sentence_odd, score_habitat_odd, label_habitat_odd, label_sentence_macro_odd, label_object,
                 label_category_odd

                 ),
        Implies(
            And(And(And(Model(x_odd, Subject_l_odd),
                        Model_person(x_odd, label_subject)), Model(y_odd, Action_l_odd)),
                And(Model_category(z_odd, label_category_odd), Model_object(z_odd, label_object))),
            And(And(And(Model(x, Subject_l), Model_person(x, label_subject)),
                    Model(y, Action_l)),
                And(Model(z, Object_l), Model_object(z, label_subject)))),
        cond_vars=[label_sentence, label_sentence_macro, label_sentence_odd, label_sentence_macro_odd],
        cond_fn=lambda t, t1, t2, t3: torch.logical_and(torch.logical_and(t.value == 1, t1.value == 0),
                                                        torch.logical_and(t2.value == 1, t3.value == 1)))

    first_axiom_negative = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 label_sentence_macro),
        And(And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                Model(y, Action_l)),
            And(Model(z, Object_l), Model_object(object, label_subject))),
        cond_vars=[label_sentence, label_sentence_macro],
        cond_fn=lambda t, t1: torch.logical_and(t.value == 0, t1.value == 0))

    second_axiom_negative = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 score_category, label_category, label_sentence_macro),
        Implies(
            And(And(And(Model(x, Subject_l),
                        Model_person(subject, label_subject)), Model(y, Action_l)),
                And(Model(z, Object_l), Model_object(object, label_subject))),
            And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                And(Model_category(score_category, label_category),
                    Model_object(object, label_subject)))),
        cond_vars=[label_sentence, label_sentence_macro],
        cond_fn=lambda t, t1: torch.logical_and(t.value == 0, t1.value == 0))

    thirth_axiom_negative = Forall(
        ltn.diag(x, Subject_l, y, Action_l, z, Object_l, subject, object,
                 label_subject, label_sentence,
                 score_habitat, label_habitat, label_sentence_macro),
        Implies(
            And(And(And(Model(x, Subject_l),
                        Model_person(subject, label_subject)), Model(y, Action_l)),
                And(Model(z, Object_l), Model_object(object, label_subject))),
            And(And(Model(x, Subject_l), Model_person(subject, label_subject)),
                And(Model_habitat(score_habitat, label_habitat),
                    Model_object(object, label_subject)))),
        cond_vars=[label_sentence, label_sentence_macro],
        cond_fn=lambda t, t1: torch.logical_and(t.value == 0, t1.value == 0))

    sat_agg = SatAgg(first_axiom_negative, first_axiom_positive, second_axiom_positive, second_axiom_negative,
                     thirth_axiom_positive, thirth_axiom_negative, fourth_axiom_positive)

    return {

        'Forall': Forall.agg_op.p,
        'Forallobject': Forall_object.agg_op.p,

        'first_axiom_positive': first_axiom_positive,
        'second_axiom_positive': second_axiom_positive,
        'thirth_axiom_positive': thirth_axiom_positive,

        'first_axiom_negative': first_axiom_negative,
        'second_axiom_negative': second_axiom_negative,
        'thirth_axiom_negative': thirth_axiom_negative,

        'fourth_axiom_positive': fourth_axiom_positive,

    }, sat_agg


def get_score_ltn(x, y):
    # print("function")
    return torch.gather(x, 1, y.view(-1, 1)).view(-1)


def get_score_contient(x, y):
    # print("function")
    return torch.sum(x * y, dim=1)


def train_ltn(dataloader, dataloader_valid, args, ndim):
    if args.log_neptune:
        run = neptune.init_run(
            project="frankissimo/ltnprobing",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTY1M2U5Ni05ZTU0LTQ0YjAtYWM0OC1jNzUyZTIwOWNiNDQifQ==")
        # your credentials

        params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
                  "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device,
                  "layer": args.layer, "train_data_path": args.train_data_path,
                  "test_data_path": args.test_data_path,
                  "model_name": args.model_name,
                  "valid_data_path": args.valid_data_path}

        run["parameters"] = params

    if args.log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Create a summary writer
        writer = SummaryWriter()

    # num_heads = 3
    # heads_per_dim = 4096
    # embed_dimension = 4096  # num_heads * heads_per_dim
    # input_dim, embed_dim, num_heads
    # attn = ScaledDotProductAttentionModel(4096,3)
    # attn = attn.to(args.probe_device)

    attn = SPOAttention(ndim).to(args.probe_device)

    # mlp = MLP(layer_sizes=(ndim, 3))
    # mlp_sentence = MLP(layer_sizes=( 12291, 1))
    mlp_person = MLP(layer_sizes=(ndim, 2))
    mlp_object = MLP(layer_sizes=(ndim, 2))
    mlp_subject_action_object = MLP(layer_sizes=(ndim, 1013 + 2))
    mlp_habitat = MLP(layer_sizes=(ndim, 28))
    mlp_continent = MLP(layer_sizes=(ndim, 9))
    mlp_category = MLP(layer_sizes=(ndim, 12))

    # mlp2 = MLP()
    # mlp3 = MLP()

    # Model = ltn.Function(func=lambda x, y: get_score_ltn(x, y))
    Model = ltn.Predicate(LogitsToPredicate(mlp_subject_action_object)).to(args.probe_device)
    # Model_continent = ltn.Function(func=lambda x, y: get_score_contient(x, y))
    # Model_habitat = ltn.Function(func=lambda x, y: get_score_contient(x, y))
    Model_habitat = ltn.Predicate(LogitsToPredicate(mlp_habitat)).to(args.probe_device)
    Model_continent = ltn.Predicate(LogitsToPredicate(mlp_continent)).to(args.probe_device)
    Model_category = ltn.Predicate(LogitsToPredicate(mlp_category)).to(args.probe_device)
    # Model_sentence = ltn.Predicate(LogitsToPredicate(mlp_sentence)).to(args.probe_device)
    Model_person = ltn.Predicate(LogitsToPredicate(mlp_person)).to(args.probe_device)
    Model_object = ltn.Predicate(LogitsToPredicate(mlp_object)).to(args.probe_device)
    # Action_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Object_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Action_model = Action_model.to(args.probe_device)
    # Object_model = Object_model.to(args.probe_device)

    parameters = []
    # parameters.extend([MatrixVector])
    parameters.extend([f for f in Model.parameters()])
    parameters.extend([f for f in Model_person.parameters()])
    parameters.extend([f for f in Model_object.parameters()])

    parameters.extend([f for f in Model_habitat.parameters()])
    parameters.extend([f for f in Model_category.parameters()])
    # parameters.extend([f for f in Action_model.parameters()])
    # parameters.extend([f for f in Object_model.parameters()])

    parameters.extend([f for f in attn.parameters()])
    # parameters.extend([f for f in attn.v_proj.parameters()])

    steps = 10

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    """
    print("loading models")
    Model.load_state_dict(torch.load("Model.pt"))
    Model_person.load_state_dict(torch.load("Model_person.pt"))
    Model_object.load_state_dict(torch.load("Model_object.pt"))
    attn.load_state_dict(torch.load("attn.pt"))
    """

    Model.train()
    Model_person.train()
    Model_object.train()
    Model_category.train()
    Model_habitat.train()
    # Action_model.eval()
    # Object_model.eval()
    attn.train()

    step = 0

    for epoch in tqdm(range(args.nr_epochs)):
        for hs, sentences, labels in dataloader:

            # forward attention
            # if args.random_baseline:
            # hs = torch.tensor(np.random.randn(hs.shape[0], hs.shape[1], hs.shape[2]), dtype=torch.float32)
            hs = hs.to(args.probe_device)
            spo, _ = attn(hs)

            subject = spo[:, 0, :]
            action = spo[:, 1, :]
            object = spo[:, 2, :]
            # scores_sub = subject.mm(attn.MatrixVector)
            # score_action = action.mm(attn.MatrixVector)
            # score_object = object.mm(attn.MatrixVector)
            # score_first = object.mm(attn.MatrixVectorFirst)
            # score_second = object.mm(attn.MatrixVectorSecond)
            # score_third = object.mm(attn.MatrixVectorThird)
            # score_continent = object.mm(attn.MatrixContinent)
            # score_habitat = object.mm(attn.MatrixHabitat)

            x = ltn.Variable("x", subject)
            y = ltn.Variable("y", action)
            z = ltn.Variable("z", object)
            score_category = ltn.Variable("score_category", object)
            score_continent = ltn.Variable("score_continent", object)
            score_habitat = ltn.Variable("score_habitat", object)
            # k2 = ltn.Variable("k2", torch.softmax(score_second, dim=1))
            # k3 = ltn.Variable("k3", torch.softmax(score_third, dim=1))
            # print(sentences)
            Subject_l = ltn.Variable("Subject_l", labels[0])
            Action_l = ltn.Variable("Action_l", torch.ones(labels[0].shape[0], dtype=torch.int64) * 1014)  # cambia
            Object_l = ltn.Variable("Object_l", labels[1])
            label_category = ltn.Variable("label_category", labels[2])
            # label_habitat = ltn.Variable("label_habitat",torch.reshape(torch.stack(labels[3:31]),(-1,28)))  # primi 03-30
            label_habitat = ltn.Variable("label_habitat", labels[3])  # primi 03-30
            # label_continent = ltn.Variable("label_continent",torch.flip(torch.reshape(torch.cat(labels[31:40]), (-1, 9)), dims=(1,)))  # primi 31-39
            label_continent = label_habitat
            label_sentence = ltn.Variable("label_sentence", labels[4])
            label_sentence_macro = ltn.Variable("label_sentence_macro", labels[5])
            # All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))

            # labels[0] = labels[0].to(args.probe_device)
            # labels[1] = labels[1].to(args.probe_device)
            # labels[2] = labels[2].to(args.probe_device)

            """
            sentence_score = ltn.Variable("sentence_score", torch.concat((subject,action,object, torch.gather(x.value,1,Subject_l.value),
                                                                           torch.gather(y.value,1,Action_l.value),
                                                                           torch.gather(z.value,1,Object_l.value)), dim=1))
            """
            # sentence_score = ltn.Variable("sentence_score",torch.concat((x.value, y.value, z.value), dim=1))

            # create axioms

            axioms, sat_agg = create_axioms(Model, Model_continent, Model_category, Model_habitat, Model_person,
                                            Model_object, Subject_l, Action_l,
                                            Object_l,
                                            score_category,
                                            score_continent, score_habitat, labels, x, y,
                                            z, label_category, label_continent, label_habitat, label_sentence,
                                            label_sentence_macro, epoch,
                                            subject, action, object, sentences)

            # calculate loss
            """
            loss = -torch.log(subject_negative.value)-torch.log(subject_positive.value)\
                   -torch.log(action_negative.value)-torch.log(action_positive.value)\
                   -torch.log(object_negative.value)-torch.log(object_positive.value)\
                   -torch.log(all_sentence_positive_implication.value)-torch.log(all_sentence_positive.value)\
                   -torch.log(all_sentence_negative.value)

            loss =  - torch.log(subject_positive.value) \
                    - torch.log(action_positive.value) \
                    - torch.log(object_positive.value)
                   #- torch.log(all_sentence_positive_implication.value) - torch.log(all_sentence_positive.value) \
                   #- torch.log(all_sentence_negative.value)
            """
            loss = 1 - sat_agg
            # - torch.log(all_sentence_positive_implication.value) - torch.log(all_sentence_positive.value) \
            # - torch.log(all_sentence_negative.value)

            """
            if epoch == 200 or epoch == 300 or epoch == 400:
                optimizer.defaults['lr'] = optimizer.defaults['lr'] / 10
            """

            # descend gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if args.log_tensorboard:
                # Add the gradient values to Tensorboard
                for name, param in attn.named_parameters():
                    writer.add_histogram(name + '/grad', param.grad, global_step=step)
                for name, param in Model.named_parameters():
                    writer.add_histogram(name + '/grad', param.grad, global_step=step)
                for name, param in Model_sentence.named_parameters():
                    writer.add_histogram(name + '/grad', param.grad, global_step=step)

            if args.log_neptune:
                for key, value in axioms.items():
                    if torch.is_tensor(value) or isinstance(value, float):
                        run[f'train/{key}'].append(value)
                    else:
                        run[f'train/{key}'].append(value.value)
                run["train/loss"].append(loss)
                run[f'train/optimizer_lr'].append(optimizer.param_groups[0]["lr"])

            step += 1
            # scheduler.step()

        if epoch % 200 == 0 and epoch != 0:
            Model.eval()
            Model_person.eval()
            Model_object.eval()
            Model_category.eval()
            Model_habitat.eval()
            attn.eval()
            os.makedirs("experiments", exist_ok=True)

            def test_part(dataloader_test, set):

                if args.log_neptune:
                    filename = "experiments" + "/" + run._sys_id + "_" + set + "_.csv"
                else:
                    filename = "experiments" + "/" + "_" + set + "_prova.csv"

                with open(filename, 'w', newline='') as file_output:
                    # create the csv writer
                    writer = csv.writer(file_output, delimiter=";")

                    row = ["sentences", "label_sentence", "first_axiom", "second_axiom", "third_axiom", "fourth_axiom",

                           "fifth_axiom",
                           "category_implication",
                           "label_sentence_even", "x_even", "subject_max_index_even", "Subject_l_even", "y_even",
                           "action_max_index_even",
                           "Action_l_even", "z_even",
                           "object_max_index_even", "Object_l_even",
                           "score_habitat_even", "habitat_index_even", "label_habitat_even", "score_category_even",
                           "category_index_even",
                           "label_category_even", "score_continent_even", "continent_index_even", "person_max_even",
                           "object_max_even",
                           "label_continent_even",

                           "label_sentence_odd", "x_odd", "subject_max_index_odd", "Subject_l_odd", "y_odd",
                           "action_max_index_odd",
                           "Action_l_odd", "z_odd",
                           "object_max_index_odd", "Object_l_odd",
                           "score_habitat_odd", "habitat_index_odd", "label_habitat_odd", "score_category_odd",
                           "category_index_odd",
                           "label_category_odd", "score_continent_odd", "continent_index_odd", "person_max_odd",
                           "object_max_odd",
                           "label_continent_odd"

                           ]

                    writer.writerow(row)
                    # self.hidden_states[idx],self.hidden_states_odd[idx], self.sentences_odd[idx], self.sentences[idx],self.labels[idx],self.labels_odd[idx]

                    with torch.no_grad():
                        for hs_even, hs_odd, sentences_even, sentences_odd, labels_even, labels_odd in tqdm(
                                dataloader_test):

                            # if args.random_baseline:
                            # hs = torch.tensor(np.random.randn(hs.shape[0], hs.shape[1], hs.shape[2]), dtype=torch.float32)
                            hs_even = hs_even.to(args.probe_device)
                            hs_odd = hs_odd.to(args.probe_device)

                            spo_even, _ = attn(hs_even)
                            spo_odd, _ = attn(hs_odd)

                            subject_even = spo_even[:, 0, :]
                            action_even = spo_even[:, 1, :]
                            object_even = spo_even[:, 2, :]

                            subject_odd = spo_odd[:, 0, :]
                            action_odd = spo_odd[:, 1, :]
                            object_odd = spo_odd[:, 2, :]

                            x_even = ltn.Variable("x_even", subject_even)
                            y_even = ltn.Variable("y_even", action_even)
                            z_even = ltn.Variable("z_even", object_even)
                            score_category_even = ltn.Variable("score_category_even", object_even)
                            score_continent_even = ltn.Variable("score_continent_even", object_even)
                            score_habitat_even = ltn.Variable("score_habitat_even", object_even)
                            Subject_l_even = ltn.Variable("Subject_l_even", labels_even[0])
                            Action_l_even = ltn.Variable("Action_l_even", torch.ones(labels_even[0].shape[0],
                                                                                     dtype=torch.int64) * 1014)  # cambia
                            Object_l_even = ltn.Variable("Object_l_even", labels_even[1])
                            label_category_even = ltn.Variable("label_category_even", labels_even[2])
                            label_habitat_even = ltn.Variable("label_habitat_even", labels_even[3])  # primi 03-30
                            # label_continent = ltn.Variable("label_continent",torch.flip(torch.reshape(torch.cat(labels[31:40]), (-1, 9)), dims=(1,)))  # primi 31-39
                            label_continent_even = label_habitat_even
                            label_sentence_even = ltn.Variable("label_sentence_even", labels_even[4])
                            label_sentence_macro_even = ltn.Variable("label_sentence_macro_even", labels_even[5])

                            x_odd = ltn.Variable("x_odd", subject_odd)
                            y_odd = ltn.Variable("y_odd", action_odd)
                            z_odd = ltn.Variable("z_odd", object_odd)
                            score_category_odd = ltn.Variable("score_category_odd", object_odd)
                            score_continent_odd = ltn.Variable("score_continent_odd", object_odd)
                            score_habitat_odd = ltn.Variable("score_habitat_odd", object_odd)

                            Subject_l_odd = ltn.Variable("Subject_l_odd", labels_odd[0])
                            Action_l_odd = ltn.Variable("Action_odd",
                                                        torch.ones(labels_odd[0].shape[0],
                                                                   dtype=torch.int64) * 1014)  # cambia
                            Object_l_odd = ltn.Variable("Object_l_odd", labels_odd[1])
                            label_category_odd = ltn.Variable("label_category_odd", labels_odd[2])
                            label_habitat_odd = ltn.Variable("label_habita_odd", labels_odd[3])  # primi 03-30
                            # label_continent = ltn.Variable("label_continent",torch.flip(torch.reshape(torch.cat(labels[31:40]), (-1, 9)), dims=(1,)))  # primi 31-39
                            label_continent_odd = label_habitat_odd
                            label_sentence_odd = ltn.Variable("label_sentence_odd", labels_odd[4])
                            label_sentence_macro_odd = ltn.Variable("label_sentence_macro_odd", labels_odd[5])

                            # create axioms
                            axioms, sat_agg = create_axioms(Model, Model_continent, Model_category, Model_habitat,
                                                            Model_person,
                                                            Model_object, Subject_l_even, Action_l_even,
                                                            Object_l_even,
                                                            score_category_even,
                                                            score_continent_even, score_habitat_even, labels, x_even,
                                                            y_even,
                                                            z_even, label_category_even, label_continent_even,
                                                            label_habitat_even,
                                                            label_sentence_even,
                                                            label_sentence_macro_even, epoch,
                                                            subject_even, action_even, object_even, sentences_even)

                            # create axioms
                            # print(sentences_even)

                            axioms_test, sat_agg_test = create_axioms_test(Model, Model_continent, Model_category,
                                                                           Model_habitat, Model_person,
                                                                           Model_object, Subject_l_even,
                                                                           Action_l_even,
                                                                           Object_l_even,
                                                                           score_category_even,
                                                                           score_continent_even, score_habitat_even,
                                                                           labels_even,
                                                                           x_even, y_even,
                                                                           z_even,

                                                                           label_category_even,
                                                                           label_continent_even,
                                                                           label_habitat_even,
                                                                           label_sentence_even,
                                                                           label_sentence_macro_even,
                                                                           epoch, subject_even, action_even,
                                                                           object_even,

                                                                           Subject_l_odd,
                                                                           Action_l_odd,
                                                                           Object_l_odd,
                                                                           score_category_odd,
                                                                           score_continent_odd, score_habitat_odd,
                                                                           labels_odd,
                                                                           x_odd, y_odd,
                                                                           z_odd,
                                                                           label_category_odd,
                                                                           label_continent_odd,
                                                                           label_habitat_odd,
                                                                           label_sentence_odd,
                                                                           label_sentence_macro_odd,
                                                                           subject_odd, action_odd, object_odd
                                                                           )

                            if args.log_neptune:
                                filename_std = "experiments/" + run._sys_id + "_" + set + "_results.csv"
                            else:
                                filename_std = "experiments/" + "_" + set + "prova2.csv"
                            if not os.path.exists(filename_std):
                                row = ["epoch"]
                                row.extend(
                                    [list(axioms.keys())[f] for f in range(len(list(axioms.keys()))) if
                                     f != 2 and f != 1])
                                row.extend([list(axioms_test.keys())[f] for f in range(len(list(axioms_test.keys()))) if
                                            f != 0 and f != 1])
                                with open(filename_std, 'a', newline="") as file_out:
                                    # create the csv writer
                                    writer_out = csv.writer(file_out, delimiter=";")

                                    writer_out.writerow(row)

                            with open(filename_std, 'a', newline="") as file_out:
                                # create the csv writer
                                writer_out = csv.writer(file_out, delimiter=";")
                                row = [epoch]
                                row.extend([float(f.value) for f in list(axioms.values()) if not isinstance(f, float)])
                                row.extend(
                                    [float(f.value) for f in list(axioms_test.values()) if not isinstance(f, float)])
                                writer_out.writerow(row)
                                file_out.flush()
                            # sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value), dim=1))
                            And = ltn.Connective(ltn.fuzzy_ops.AndMin())
                            Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
                            Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

                            torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
                            # print(sentences_even)
                            positive_index = [f for f in range(len(labels[5])) if labels[5][f] == 1][:10]

                            x_even.value = Model.model(x_even.value)
                            y_even.value = Model.model(y_even.value)

                            x_odd.value = Model.model(x_odd.value)
                            y_odd.value = Model.model(y_odd.value)

                            score_category_even.value = Model_category.model(z_even.value)
                            score_habitat_even.value = Model_habitat.model(z_even.value)
                            score_continent_even.value = Model_continent.model(z_even.value)

                            score_category_odd.value = Model_category.model(z_odd.value)
                            score_habitat_odd.value = Model_habitat.model(z_odd.value)
                            score_continent_odd.value = Model_continent.model(z_odd.value)
                            z_odd.value = Model.model(z_odd.value)

                            subject_max_even = torch.max(x_even.value, dim=1)[0].cpu().detach().view(-1, 1)
                            action_max_even = torch.max(y_even.value, dim=1)[0].cpu().detach().view(-1, 1)
                            z_even.value = Model.model(z_even.value)
                            object_max_even = torch.max(z_even.value, dim=1)[0].cpu().detach().view(-1, 1)

                            subject_max_odd = torch.max(x_odd.value, dim=1)[0].cpu().detach().view(-1, 1)
                            action_max_odd = torch.max(y_odd.value, dim=1)[0].cpu().detach().view(-1, 1)
                            object_max_odd = torch.max(z_odd.value, dim=1)[0].cpu().detach().view(-1, 1)

                            score_subject_gt_even = torch.gather(x_even.value.cpu().detach(), 1,
                                                                 labels_even[0].view(-1, 1))
                            score_object_gt_even = torch.gather(z_even.value.cpu().detach(), 1,
                                                                labels_even[1].view(-1, 1))
                            score_category_gt_even = torch.gather(score_category_even.value.cpu().detach(), 1,
                                                                  labels_even[2].view(-1, 1))
                            score_habitat_gt_even = torch.gather(score_habitat_even.value.cpu().detach(), 1,
                                                                 labels_even[3].view(-1, 1))

                            score_subject_gt_odd = torch.gather(x_odd.value.cpu().detach(), 1,
                                                                labels_odd[0].view(-1, 1))
                            score_object_gt_odd = torch.gather(z_odd.value.cpu().detach(), 1, labels_odd[1].view(-1, 1))
                            score_category_gt_odd = torch.gather(score_category_odd.value.cpu().detach(), 1,
                                                                 labels_odd[2].view(-1, 1))
                            # score_habitat_gt_odd = torch.gather(score_habitat_odd.value.cpu().detach(), 1, labels_odd[3].view(-1, 1))

                            category = torch.max(score_category_even.value, dim=1)[0]
                            habitat = torch.max(score_habitat_even.value, dim=1)[0]

                            subject_max_index_even = torch.argmax(x_even.value, dim=1)
                            action_max_index_even = torch.argmax(y_even.value, dim=1)
                            object_max_index_even = torch.argmax(z_even.value, dim=1)
                            # nation_max = torch.max(k.value, dim=1)[0]
                            # nation_selected = torch.gather(k.value, 1, Nation_l.value).view(-1)
                            person_max_even = Model_person.model(subject_even)[:, 1]
                            object_person_even = Model_object.model(object_even)[:, 1]
                            habitat_index_even = torch.argmax(score_habitat_even.value, dim=1)
                            category_index_even = torch.argmax(score_category_even.value, dim=1)
                            continent_index_even = torch.argmax(score_continent_even.value, dim=1)

                            category_odd = torch.max(score_category_odd.value, dim=1)[0]
                            habitat_odd = torch.max(score_habitat_odd.value, dim=1)[0]

                            subject_max_index_odd = torch.argmax(x_odd.value, dim=1)
                            action_max_index_odd = torch.argmax(y_odd.value, dim=1)
                            object_max_index_odd = torch.argmax(z_odd.value, dim=1)
                            # nation_max = torch.max(k.value, dim=1)[0]
                            # nation_selected = torch.gather(k.value, 1, Nation_l.value).view(-1)
                            person_max_odd = Model_person.model(subject_odd)[:, 1]
                            object_person_odd = Model_object.model(object_odd)[:, 1]
                            habitat_index_odd = torch.argmax(score_habitat_odd.value, dim=1)
                            category_index_odd = torch.argmax(score_category_odd.value, dim=1)
                            continent_index_odd = torch.argmax(score_continent_odd.value, dim=1)

                            # TEST FOR MODEL KNOWLEDGE
                            # first_axiom= Gordon è un cane

                            first_axiom = torch.min(
                                torch.min(torch.min(score_subject_gt_even, person_max_even.cpu().detach().view(-1, 1)),
                                          action_max_even),
                                torch.min(score_object_gt_even, object_person_even.cpu().detach().view(-1, 1)))
                            second_axiom = torch.min(
                                torch.min(score_subject_gt_even, person_max_even.cpu().detach().view(-1, 1), ),
                                torch.min(score_subject_gt_even, score_category_gt_even))
                            third_axiom = (1. - first_axiom) + torch.mul(first_axiom, second_axiom)

                            fourth_axiom = torch.min(
                                torch.min(score_subject_gt_even, person_max_even.cpu().detach().view(-1, 1)),
                                score_habitat_gt_even)
                            fifth_axiom = (1. - first_axiom) + torch.mul(first_axiom, fourth_axiom)

                            first_axiom_b = torch.min(
                                torch.min(torch.min(score_subject_gt_odd, person_max_odd.cpu().detach().view(-1, 1)),
                                          action_max_odd),
                                torch.min(score_category_gt_odd, object_person_odd.cpu().detach().view(-1, 1)))

                            category_implication = (1. - first_axiom_b) + torch.mul(first_axiom, first_axiom_b)

                            # Test for correct sentences

                            print("saving csv")
                            for f in range(len(sentences_even)):
                                row = []
                                row.append(sentences_even[f])
                                row.append(label_sentence_even.value[f].item())

                                row.append(first_axiom[f].item())
                                row.append(second_axiom[f].item())
                                row.append(third_axiom[f].item())
                                row.append(fourth_axiom[f].item())
                                row.append(fifth_axiom[f].item())
                                row.append(category_implication[f].item())
                                # row.append(sixty_axiom[f].item())
                                row.append(label_sentence_even.value[f].item())
                                row.append(max(x_even.value[f]).item())
                                row.append(subject_max_index_even[f].item())
                                row.append(Subject_l_even.value[f].item())
                                row.append(max(y_even.value[f]).item())
                                row.append(action_max_index_even[f].item())
                                row.append(Action_l_even.value[f].item())
                                row.append(max(z_even.value[f]).item())
                                row.append(object_max_index_even[f].item())
                                row.append(Object_l_even.value[f].item())
                                row.append(max(score_habitat_even.value[f]).item())
                                row.append(habitat_index_even[f].item())
                                row.append(label_habitat_even.value[f].item())
                                row.append(max(score_category_even.value[f]).item())
                                row.append(category_index_even[f].item())
                                row.append(label_category_even.value[f].item())
                                row.append(max(score_continent_even.value[f]).item())
                                row.append(continent_index_even[f].item())
                                row.append(person_max_even.cpu().detach().view(-1, 1)[f].item())
                                row.append(object_max_even.cpu().detach().view(-1, 1)[f].item())
                                row.append(label_continent_even.value[f].item())
                                writer.writerow(row)

                            if args.log_neptune:
                                for key, value in axioms.items():
                                    if torch.is_tensor(value) or isinstance(value, float):
                                        run[f'{set}/{key}'].append(value)
                                    else:
                                        run[f'{set}/{key}'].append(value.value)

                                for key, value in axioms_test.items():
                                    if torch.is_tensor(value) or isinstance(value, float):
                                        run[f'{set}/{key}'].append(value)
                                    else:
                                        run[f'{set}/{key}'].append(value.value)

            test_part(dataloader_valid, "val")

            Model.train()
            Model_person.train()
            Model_object.train()
            Model_category.train()
            Model_habitat.train()
            attn.train()

            os.makedirs(run._sys_id, exist_ok=True)
            torch.save(Model.state_dict(), run._sys_id + "/Model.pt")
            torch.save(Model_category.state_dict(), run._sys_id + "/Model_category.pt")
            torch.save(Model_person.state_dict(), run._sys_id + "/Model_person.pt")
            torch.save(Model_object.state_dict(), run._sys_id + "/Model_object.pt")
            torch.save(attn.state_dict(), run._sys_id + "/attn.pt")

            # test
            # test

            if epoch == (args.nr_epochs - 1):
                del dataloader_valid
                del dataloader

                dataset_test, _ = get_dataset(None, args.test_data_path)

                if not args.random_baseline:
                    generation_args.data_path = args.test_data_path
                    hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))
                else:

                    generation_args.data_path = args.test_data_path
                    hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))
                    hs_test = np.random.randn(hs_test.shape[0], hs_test.shape[1], hs_test.shape[2],
                                              hs_test.shape[3],
                                              hs_test.shape[4])

                if args.random_label:
                    zero = max([f[0] for f in dataset_test['labels']])
                    first = max([f[1] for f in dataset_test['labels']])
                    second = max([f[2] for f in dataset_test['labels']])
                    third = max([f[3] for f in dataset_test['labels']])
                    fourth = max([f[4] for f in dataset_test['labels']])
                    fifth = max([f[5] for f in dataset_test['labels']])

                    random_label_test = [[random.randint(0, zero), random.randint(0, first), random.randint(0, second),
                                           random.randint(0, third), random.randint(0, fourth),
                                           random.randint(0, fifth)]
                                          for f in dataset_test['labels']]

                    dataset_test.remove_columns("labels").add_column("labels", random_label_test)

                hs_test_t = torch.Tensor(hs_test).squeeze()
                batch, nr_tokens, ndim = hs_test_t.shape
                hs_dataset_test = CustomDatasetTest(hs_test_t, dataset_test)
                batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_test_t)
                hs_dataloader_test = DataLoader(hs_dataset_test, batch_size=batch_size)
                test_part(hs_dataloader_test, "test")

    if args.log_neptune:
        run.stop()


def main(args, generation_args):
    print(f'CUDA available? {torch.cuda.is_available()}')
    print(f'Nr. of CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Name of first device: {torch.cuda.get_device_name(0)}')
    print(f'Random seed torch device: {torch.random.initial_seed()}')
    # torch.manual_seed(0)

    dataset_train, _ = get_dataset(None, args.train_data_path)

    dataset_valid, _ = get_dataset(None, args.valid_data_path)

    # load dataset and hidden states
    # generation_args.data_path = args.train_data_path
    # hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
    # generation_args.data_path = args.test_data_path
    # hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))

    if not args.random_baseline:
        generation_args.data_path = args.train_data_path
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))

        generation_args.data_path = args.valid_data_path
        hs_valid = trim_hidden_states(load_single_generation(vars(generation_args)))
    else:
        generation_args.data_path = args.train_data_path
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
        generation_args.data_path = args.valid_data_path
        hs_valid = trim_hidden_states(load_single_generation(vars(generation_args)))
        hs_train = np.random.randn(hs_train.shape[0], hs_train.shape[1], hs_train.shape[2], hs_train.shape[3],
                                   hs_train.shape[4])
        hs_valid = np.random.randn(hs_valid.shape[0], hs_valid.shape[1], hs_valid.shape[2], hs_valid.shape[3],
                                   hs_valid.shape[4])
    if args.random_label:
        print("----- random labels -----")
        zero = max([f[0] for f in dataset_train['labels']])
        first = max([f[1] for f in dataset_train['labels']])
        second = max([f[2] for f in dataset_train['labels']])
        third = max([f[3] for f in dataset_train['labels']])
        fourth = max([f[4] for f in dataset_train['labels']])
        fifth = max([f[5] for f in dataset_train['labels']])

        random_label_training = [[random.randint(0, zero), random.randint(0, first), random.randint(0, second),
                                  random.randint(0, third), random.randint(0, fourth), random.randint(0, fifth)]
                                 for f in dataset_train['labels']]

        zero = max([f[0] for f in dataset_valid['labels']])
        first = max([f[1] for f in dataset_valid['labels']])
        second = max([f[2] for f in dataset_valid['labels']])
        third = max([f[3] for f in dataset_valid['labels']])
        fourth = max([f[4] for f in dataset_valid['labels']])
        fifth = max([f[5] for f in dataset_valid['labels']])

        random_label_valid = [[random.randint(0, zero), random.randint(0, first), random.randint(0, second),
                               random.randint(0, third), random.randint(0, fourth), random.randint(0, fifth)]
                              for f in dataset_valid['labels']]

        dataset_train.remove_columns("labels").add_column("labels", random_label_training)
        dataset_valid.remove_columns("labels").add_column("labels", random_label_valid)

    with open('dict_wordnet_16_09_23.pickle', 'rb') as handle:
        my_dict_wordnet = pickle.load(handle)

    # train LTN probe
    hs_train_t = torch.Tensor(hs_train).squeeze()
    hs_dataset_train = CustomDataset(hs_train_t, dataset_train)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_train_t)
    hs_dataloader_train = DataLoader(hs_dataset_train, batch_size=batch_size, shuffle=True)
    val = [f[0] for f in dataset_train["labels"]]
    val.extend([f[1] for f in dataset_train["labels"]])
    val.extend([f[2] for f in dataset_train["labels"]])
    # val.extend([f[2] for f in dataset_train["labels"]])
    # val.extend([f[3] for f in dataset_train["labels"]]) #nations

    # content_unique 9 elementi
    # habitat_unique 28

    print("tot subjects", len(list(set([f[0] for f in dataset_train["labels"]]))))
    print("tot objects", len(list(set([f[1] for f in dataset_train["labels"]]))))
    print("tot first_category", len(list(set([f[2] for f in dataset_train["labels"]]))))
    print("tot  continent_unique", len(my_dict_wordnet['continenti_unique']))
    print("tot  habita_unique", len(my_dict_wordnet['habita_unique']))
    # print("tot second_category", len(list(set([f[3] for f in dataset_train["labels"]]))))
    # print("tot third_category", len(list(set([f[4] for f in dataset_train["labels"]]))))
    # print("tot objects_category", len(list(set([f[5] for f in dataset_train["labels"]]))))

    print("max_elements_objects", max(list(set([f[1] for f in dataset_train["labels"]]))))
    print("max_elements_subjects", max(list(set([f[0] for f in dataset_train["labels"]]))))
    print("max_elements_first_category", max(list(set([f[2] for f in dataset_train["labels"]]))))
    # print("max_elements_second_category", max(list(set([f[3] for f in dataset_train["labels"]]))))
    # print("max_elements_third_category", max(list(set([f[4] for f in dataset_train["labels"]]))))
    # print("max_elements_fourty_category", max(list(set([f[5] for f in dataset_train["labels"]]))))

    hs_valid_t = torch.Tensor(hs_valid).squeeze()
    batch, nr_tokens, ndim = hs_valid_t.shape
    hs_dataset_valid = CustomDatasetTest(hs_valid_t, dataset_valid)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_valid_t)
    hs_dataloader_valid = DataLoader(hs_dataset_valid, batch_size=batch_size)

    train_ltn(hs_dataloader_train, hs_dataloader_valid, args, ndim)


if __name__ == "__main__":
    parser = get_parser()
    generation_args, _ = parser.parse_known_args()
    # We'll also add some additional args for evaluation
    parser.add_argument("--nr_epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--log_tensorboard", action='store_true')
    parser.add_argument("--train_data_path", type=str, default='test_dataset_16_09_23.txt')
    parser.add_argument("--test_data_path", type=str, default='test_dataset_16_09_23.txt')
    parser.add_argument("--valid_data_path", type=str, default='test_dataset_16_09_23.txt')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    parser.add_argument("--random_label", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    args = parser.parse_args()
    main(args, generation_args)
