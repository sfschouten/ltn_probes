import math
import os
import pickle

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ltn

from utils import get_parser, load_single_generation, get_dataset
from customdataloader import CustomDataset

from dotenv import load_dotenv

load_dotenv()


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
        self.MatrixVector = nn.Parameter(2e-4 * torch.rand([4096, 1013 + 2]), requires_grad=True)
        self.MatrixVectorFirst = nn.Parameter(2e-4 * torch.rand([4096, 12]), requires_grad=True)
        # self.MatrixVectorSecond = nn.Parameter(2e-4 * torch.rand([4096, 43]), requires_grad=True)
        # self.MatrixVectorThird = nn.Parameter(2e-4 * torch.rand([4096, 156]), requires_grad=True)
        self.MatrixContinent = nn.Parameter(2e-4 * torch.rand([4096, 9]), requires_grad=False)
        self.MatrixHabitat = nn.Parameter(2e-4 * torch.rand([4096, 28]), requires_grad=True)

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
        probs = self.sigmoid(logits)
        if l != None:
            out = torch.sum(probs * l, dim=1)
        else:
            out = probs
        return out.view(-1)


def create_axioms(Model, Model_continent, Model_person, Subject_l, Action_l, Object_l, score_category, score_continent,
                  score_habitat, labels, x, y,
                  z, label_category, label_continent, label_habitat, label_sentence, epoch):
    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall_person = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6.0), quantifier="f")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
    if epoch < 100:
        # Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=2.0))
    if epoch >= 100:
        # Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=4.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=2.0))
    if epoch >= 150:
        # Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=2.0))
    """
    if epoch >= 500:
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=7.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=7.0))
    if epoch >= 700:
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=8.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=8.0))
    if epoch >= 900:
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=9.0), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=9.0))
    """





    """
    subject_negative = Forall(ltn.diag(label_sentence, x, Subject_l),
                              Not(Model(x, Subject_l)),
                              cond_vars=[label_sentence],
                              cond_fn=lambda t: t.value == 0)
    """



    subject_positive = Forall(ltn.diag(label_sentence, x, Subject_l),
                              Model(x, Subject_l),
                              cond_vars=[label_sentence],
                              cond_fn=lambda t: t.value == 1)

    action_positive = Forall(ltn.diag(y, Action_l), Model(y, Action_l))
    
    axiom_category = Forall(ltn.diag(label_sentence, score_category, label_category),
                        Model(score_category, label_category),
                        cond_vars=[label_sentence],
                        cond_fn=lambda t: t.value == 1)

    


    object_positive = Forall(ltn.diag(label_sentence, z, Object_l),
                             Model(z, Object_l),
                             cond_vars=[label_sentence],
                             cond_fn=lambda t: t.value == 1)



    """
    list_continents=[]
    for item in range(label_continent.value.shape[1]):
        #label_continent.value

        new_continent_variable=ltn.Variable("new_continent_variable",label_continent.value[:,item])
        score_continent_variable=ltn.Variable("score_continent_variable",score_continent.value[:,item])

        axiom_continent = Forall(ltn.diag(label_sentence, score_continent_variable, new_continent_variable),
                                 Model_continent(score_continent_variable, new_continent_variable),
                                 cond_vars=[label_sentence,new_continent_variable],
                                 cond_fn=lambda t,t1: torch.logical_and(t.value == 1 , t1.value==1))
        list_continents.append(axiom_continent)

    axiom_continent = SatAgg(list_continents[0],list_continents[1],list_continents[2],list_continents[3],list_continents[4],list_continents[5],
                             list_continents[6],list_continents[7],list_continents[8])
    """

    axiom_habitat = Forall(ltn.diag(label_sentence, score_habitat, label_habitat),
                           Model(score_habitat, label_habitat),
                           cond_vars=[label_sentence],
                           cond_fn=lambda t: t.value == 1)


    """

    is_not_person_subject = Forall(ltn.diag(label_sentence, x),
                                   Not(Model_person(x)),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 0)

    is_not_person_object = Forall(ltn.diag(label_sentence, z),
                                  Not(Model_person(z)),
                                  cond_vars=[label_sentence],
                                  cond_fn=lambda t: t.value == 1)

    is_person_subject = Forall(ltn.diag(label_sentence, x),
                               Model_person(x),
                               cond_vars=[label_sentence],
                               cond_fn=lambda t: t.value == 1)

    is_person_object = Forall(ltn.diag(label_sentence, z),
                              Model_person(z),
                              cond_vars=[label_sentence],
                              cond_fn=lambda t: t.value == 0)
    """

    sat_agg = SatAgg(axiom_habitat,subject_positive, object_positive, action_positive, axiom_category)
    #subject_positive, object_positive, action_positive, axiom_category,
    #,is_not_person_subject, is_person_subject, is_not_person_object, is_person_object
    #
    # sat_agg = torch.log(object_positive.value) #+ torch.log(action_positive.value) + torch.log(subject_positive.value)

    # sat_agg = SatAgg(subject_positive, action_positive, object_positive,
    #                  subject_negative, action_negative, object_negative, all_sentence_positive, all_sentence_negative)

    return {
       'subject_positive': subject_positive,
        # 'subject_negative': subject_negative,
        # 'is_person_subject': is_person_subject,
        # 'is_person_object': is_person_object,
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
        # 'is_person_subject': is_person_subject,
        # 'is_not_person_subject': is_not_person_subject,
        # 'is_person_object': is_person_object,
        # 'is_not_person_object': is_not_person_object,

        # 'all_sentence_positive_implication': all_sentence_positive_implication,
        # 'all_sentence_negative_implication': all_sentence_negative_implication,
    }, sat_agg


def get_score_ltn(x, y):
    # print("function")
    return torch.gather(x, 1, y.view(-1, 1)).view(-1)


def get_score_contient(x, y):
    # print("function")
    return torch.sum(x * y, dim=1)


def train_ltn(dataloader, dataloader_test, args, ndim):
    if args.log_neptune:
        import neptune
        run = neptune.init_run(
            project="frankissimo/ltnprobing",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTY1M2U5Ni05ZTU0LTQ0YjAtYWM0OC1jNzUyZTIwOWNiNDQifQ==")
        # your credentials

        params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
                  "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device}
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
    mlp_person = MLP(layer_sizes=(1015, 1))

    # mlp2 = MLP()
    # mlp3 = MLP()

    Model = ltn.Function(func=lambda x, y: get_score_ltn(x, y))
    Model_continent = ltn.Function(func=lambda x, y: get_score_contient(x, y))
    Model_habitat = ltn.Function(func=lambda x, y: get_score_contient(x, y))
    # Model_sentence = ltn.Predicate(LogitsToPredicate(mlp_sentence)).to(args.probe_device)
    Model_person = ltn.Predicate(LogitsToPredicate(mlp_person)).to(args.probe_device)
    # Action_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Object_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Action_model = Action_model.to(args.probe_device)
    # Object_model = Object_model.to(args.probe_device)

    parameters = []
    # parameters.extend([MatrixVector])
    # parameters.extend([f for f in Model_person.parameters()])
    # parameters.extend([f for f in Action_model.parameters()])
    # parameters.extend([f for f in Object_model.parameters()])

    parameters.extend([f for f in attn.parameters()])
    # parameters.extend([f for f in attn.v_proj.parameters()])

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0)

    Model.train()
    Model_person.train()
    # Action_model.eval()
    # Object_model.eval()
    attn.train()

    step = 0
    for epoch in tqdm(range(args.nr_epochs)):
        for hs, _, labels in dataloader:

            # forward attention
            # if args.random_baseline:
            # hs = torch.tensor(np.random.randn(hs.shape[0], hs.shape[1], hs.shape[2]), dtype=torch.float32)
            hs = hs.to(args.probe_device)
            spo, _ = attn(hs)

            subject = spo[:, 0, :]
            action = spo[:, 1, :]
            object = spo[:, 2, :]
            scores_sub = subject.mm(attn.MatrixVector)
            score_action = action.mm(attn.MatrixVector)
            score_object = object.mm(attn.MatrixVector)
            score_first = object.mm(attn.MatrixVectorFirst)
            # score_second = object.mm(attn.MatrixVectorSecond)
            # score_third = object.mm(attn.MatrixVectorThird)
            score_continent = object.mm(attn.MatrixContinent)
            score_habitat = object.mm(attn.MatrixHabitat)

            x = ltn.Variable("x", torch.softmax(scores_sub, dim=1))
            y = ltn.Variable("y", torch.softmax(score_action, dim=1))
            z = ltn.Variable("z", torch.softmax(score_object, dim=1))
            score_category = ltn.Variable("score_category", torch.softmax(score_first, dim=1))
            score_continent = ltn.Variable("score_continent", torch.sigmoid(score_continent))
            score_habitat = ltn.Variable("score_habitat", torch.softmax(score_habitat,dim=1))
            # k2 = ltn.Variable("k2", torch.softmax(score_second, dim=1))
            # k3 = ltn.Variable("k3", torch.softmax(score_third, dim=1))

            Subject_l = ltn.Variable("Subject_l", labels[0])
            Action_l = ltn.Variable("Action_l", torch.ones(labels[0].shape[0], dtype=torch.int64) * 1014)  # cambia
            Object_l = ltn.Variable("Object_l", labels[1])
            label_category = ltn.Variable("label_category", labels[2])
            #label_habitat = ltn.Variable("label_habitat",torch.reshape(torch.stack(labels[3:31]),(-1,28)))  # primi 03-30
            label_habitat = ltn.Variable("label_habitat",labels[3])  # primi 03-30
            #label_continent = ltn.Variable("label_continent",torch.flip(torch.reshape(torch.cat(labels[31:40]), (-1, 9)), dims=(1,)))  # primi 31-39
            label_continent = label_habitat
            label_sentence = ltn.Variable("label_sentence", labels[5])
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
            axioms, sat_agg = create_axioms(Model, Model_continent, Model_person, Subject_l, Action_l, Object_l,
                                            score_category,
                                            score_continent, score_habitat, labels, x, y,
                                            z, label_category, label_continent, label_habitat, label_sentence, epoch)

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
                    if torch.is_tensor(value):
                        run[f'train/{key}'].append(value)
                    else:
                        run[f'train/{key}'].append(value.value)
                run["train/loss"].append(loss)

            step += 1

        if epoch % 10 == 0:
            Model.eval()
            Model_person.eval()
            # Action_model.eval()
            # Object_model.eval()
            attn.eval()

            with torch.no_grad():
                for hs, sentences, labels in tqdm(dataloader_test):

                    # if args.random_baseline:
                    # hs = torch.tensor(np.random.randn(hs.shape[0], hs.shape[1], hs.shape[2]), dtype=torch.float32)
                    hs = hs.to(args.probe_device)
                    spo, _ = attn(hs)
                    subject = spo[:, 0, :]
                    action = spo[:, 1, :]
                    object = spo[:, 2, :]
                    scores_sub = subject.mm(attn.MatrixVector)
                    score_action = action.mm(attn.MatrixVector)
                    score_object = object.mm(attn.MatrixVector)
                    score_first = object.mm(attn.MatrixVectorFirst)
                    # score_second = object.mm(attn.MatrixVectorSecond)
                    # score_third = object.mm(attn.MatrixVectorThird)
                    score_continent = object.mm(attn.MatrixContinent)
                    score_habitat = object.mm(attn.MatrixHabitat)

                    x = ltn.Variable("x", torch.softmax(scores_sub, dim=1))
                    y = ltn.Variable("y", torch.softmax(score_action, dim=1))
                    z = ltn.Variable("z", torch.softmax(score_object, dim=1))
                    score_category = ltn.Variable("score_category", torch.softmax(score_first, dim=1))
                    score_continent = ltn.Variable("score_continent", torch.sigmoid(score_continent))
                    score_habitat = ltn.Variable("score_habitat", torch.sigmoid(score_habitat))
                    # k2 = ltn.Variable("k2", torch.softmax(score_second, dim=1))
                    # k3 = ltn.Variable("k3", torch.softmax(score_third, dim=1))

                    Subject_l = ltn.Variable("Subject_l", labels[0])
                    Action_l = ltn.Variable("Action_l",
                                            torch.ones(labels[0].shape[0], dtype=torch.int64) * 1014)  # cambia
                    Object_l = ltn.Variable("Object_l", labels[1])
                    label_category = ltn.Variable("label_category", labels[2])
                    # label_habitat = ltn.Variable("label_habitat",torch.reshape(torch.stack(labels[3:31]),(-1,28)))  # primi 03-30
                    label_habitat = ltn.Variable("label_habitat", labels[3])  # primi 03-30
                    # label_continent = ltn.Variable("label_continent",torch.flip(torch.reshape(torch.cat(labels[31:40]), (-1, 9)), dims=(1,)))  # primi 31-39
                    label_continent = label_habitat
                    label_sentence = ltn.Variable("label_sentence", labels[5])
                    # All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))

                    # labels[0] = labels[0].to(args.probe_device)
                    # labels[1] = labels[1].to(args.probe_device)
                    # labels[2] = labels[2].to(args.probe_device)

                    ltn.Variable("sentence_score",
                                 torch.concat((subject, action, object, torch.gather(x.value, 1, Subject_l.value),
                                               torch.gather(y.value, 1, Action_l.value),
                                               torch.gather(z.value, 1, Object_l.value)), dim=1))

                    # create axioms
                    axioms, sat_agg = create_axioms(Model, Model_continent, Model_person, Subject_l, Action_l, Object_l,
                                                    score_category,
                                                    score_continent, score_habitat, labels, x, y,
                                                    z, label_category, label_continent, label_habitat, label_sentence,
                                                    epoch)

                    """
                    sentence_score = ltn.Variable("sentence_score",
                                                  torch.concat((torch.gather(x.value, 1, torch.argmax(x.value,dim=1).view(-1, 1)).view(-1, 1)
                                                                , torch.gather(y.value, 1, torch.argmax(y.value,dim=1).view(-1, 1)).view(-1, 1)
                                                                , torch.gather(z.value, 1, torch.argmax(z.value,dim=1).view(-1, 1)).view(-1, 1)),
                                                               dim=1))
                    """
                    # sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value), dim=1))
                    And = ltn.Connective(ltn.fuzzy_ops.AndMin())
                    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
                    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

                    torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
                    print(sentences)
                    positive_index = [f for f in range(len(labels[5])) if labels[5][f] == 1][:10]
                    print(f" SUBJECT                      ",
                          [torch.argmax(x.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [labels[0].cpu().detach().numpy()[f1] for f1 in positive_index])
                    # print(f"(LIVE) IN SENTENCE                   ", torch.argmax(y.value, dim=1),torch.ones(labels[0].shape[0], dtype=torch.int64) * 1014)
                    print(f" OBJECT                   ",
                          [torch.argmax(z.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [labels[1].cpu().detach().numpy()[f1] for f1 in positive_index])
                    print(f" FirstCategory                   ",
                          [torch.argmax(score_category.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [label_category.value.cpu().detach().numpy()[f1][0] for f1 in positive_index])

                    """

                    print(f" Continent                   ",
                          [torch.argmax(score_continent.value, dim=1).cpu().detach().numpy()[f] for f in
                           positive_index],
                          [label_continent.value.cpu().detach().numpy()[f1][0] for f1 in positive_index])
                    """

                    print(f" Habitat                   ",
                          [torch.argmax(score_habitat.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [label_habitat.value.cpu().detach().numpy()[f1][0] for f1 in positive_index])

                    """
                    print(f" SeconCategory                   ",
                          [torch.argmax(k2.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [label_second.value.cpu().detach().numpy()[f1][0] for f1 in positive_index])
                    print(f" ThirdCategory                   ",
                          [torch.argmax(k3.value, dim=1).cpu().detach().numpy()[f] for f in positive_index],
                          [label_third.value.cpu().detach().numpy()[f1][0] for f1 in positive_index])
                    """
                    print(f" PERSON - subject - prediction                     ", Model_person(x))
                    print(f" PERSON - subject - label                     ", labels[5])
                    print(f" PERSON - object - prediction                     ", Model_person(z))
                    print(f" PERSON - object - label                     ", labels[5])

                    label_subject = ltn.Variable("subject_label", torch.argmax(x.value, dim=1))
                    label_action = ltn.Variable("subject_action", torch.argmax(y.value, dim=1))
                    label_object = ltn.Variable("subject_object", torch.argmax(z.value, dim=1))
                    # label_object = ltn.Variable("subject_nation", torch.argmax(k.value, dim=1))

                    label_sentence = ltn.Variable("label_d", labels[4].clone().detach())  # sentence correctly
                    # label_person_subject = ltn.Variable("label_person_subject", labels[4].clone().detach())
                    # label_person_action = ltn.Variable("label_person_action", labels[5].clone().detach())
                    # label_person_object = ltn.Variable("label_person_object", labels[6].clone().detach())

                    subject_max = torch.max(x.value, dim=1)[0]
                    action_max = torch.max(y.value, dim=1)[0]
                    object_max = torch.max(z.value, dim=1)[0]
                    # nation_max = torch.max(k.value, dim=1)[0]
                    # nation_selected = torch.gather(k.value, 1, Nation_l.value).view(-1)
                    person_max = Model_person(x).value
                    object_person = Model_person(z).value

                    """

                    subject_positive = Forall(
                        ltn.diag(x, label_subject, y, label_action, z, label_object, label_sentence),
                        And(And(And(Model(x, label_subject), Model_person(x)),
                                Model(y, label_action)),
                            And(Model(z, label_object), Not(Model_person(z)))),
                        cond_vars=[label_sentence],
                        cond_fn=lambda t: t.value == 1)

                    subject_negative = Forall(
                        ltn.diag(x, label_subject, y, label_action, z, label_object, label_sentence),
                        And(And(And(Model(x, label_subject), Model_person(x)),
                                Model(y, label_action)),
                            And(Model(z, label_object), Not(Model_person(z)))),
                        cond_vars=[label_sentence],
                        cond_fn=lambda t: t.value == 0)

                    # print(f" Sentences positive                  ", subject_positive.value)
                    # print(f" Sentences negative                  ", subject_negative.value)

                    # print("Sentences positive - for each row", torch.min(torch.min(torch.min(subject_max, person_max), action_max), torch.min(object_max, 1 - object_person)), labels[4])

                    # peter and  person and is live in and nation and not person nation
                    """
                    """
                    print("Sentences positive nation - for each row",
                          torch.min(torch.min(torch.min(subject_max, person_max), action_max),
                                    torch.min(nation_max, 1 - object_person)), labels[4])
                    """
                    """
                    ones = torch.ones_like(x)
                    return torch.minimum(1. - x + y, ones)
        
                    """
                    first_sentence = torch.min(torch.min(torch.min(subject_max, person_max), action_max),
                                               torch.min(object_max, 1 - object_person))

                    # second_sentence = torch.min(torch.min(torch.min(subject_max, person_max), action_max),torch.min(nation_selected, 1 - object_person))

                    # ones = torch.ones_like(first_sentence)
                    # implication= torch.minimum(1. - first_sentence + second_sentence, ones)
                    # implication = 1. - first_sentence + torch.mul(first_sentence, second_sentence)

                    positive_sentence = torch.min(torch.min(torch.min(subject_max, person_max), action_max),
                                                  torch.min(object_max, 1 - object_person))

                    # implication2 = 1. - torch.min(first_sentence, positive_sentence) + torch.mul(torch.min(first_sentence, positive_sentence), second_sentence)

                    # print("Sentences positive nation city then nation  - for each row", implication, labels[4])
                    # print("Sentences positive nation city then nation  with sentence positive - for each row", implication2, labels[4])

                    # print(f"'PETER LIVES IN AMSTERDAM' = SENTENCE", Model_sentence(sentence_score).value, labels[3])
                    # print(attention_values)

                    if args.log_neptune:
                        for key, value in axioms.items():
                            if torch.is_tensor(value):
                                run[f'test/{key}'].append(value)
                            else:
                                run[f'test/{key}'].append(value.value)

            Model.train()
            Model_person.train()
            attn.train()

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
    dataset_test, _ = get_dataset(None, args.test_data_path)

    def trim_hidden_states(hs):
        mask = np.isfinite(hs)  # (nr_samples, nr_layers, nr_tokens, nr_dims)
        mask = mask.all(axis=3)
        token_cnt = mask.sum(axis=2)
        trim_i = token_cnt.max()
        print(f'Trimming to {trim_i} from {hs.shape[2]}.')
        return hs[:, :, :trim_i, :]

    # load dataset and hidden states
    generation_args.data_path = args.train_data_path
    hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
    generation_args.data_path = args.test_data_path
    hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))

    if not args.random_baseline:
        generation_args.data_path = args.train_data_path
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
        generation_args.data_path = args.test_data_path
        hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))
    else:
        generation_args.data_path = args.train_data_path
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
        generation_args.data_path = args.test_data_path
        hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))
        hs_train = np.random.randn(hs_train.shape[0], hs_train.shape[1], hs_train.shape[2], hs_train.shape[3],
                                   hs_train.shape[4])
        hs_test = np.random.randn(hs_test.shape[0], hs_test.shape[1], hs_test.shape[2], hs_test.shape[3],
                                  hs_test.shape[4])

    with open('dict_wordnet_24_08_23.pickle', 'rb') as handle:
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

    # test
    hs_test_t = torch.Tensor(hs_test).squeeze()
    batch, nr_tokens, ndim = hs_test_t.shape
    hs_dataset_test = CustomDataset(hs_test_t, dataset_test)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_test_t)
    hs_dataloader_test = DataLoader(hs_dataset_test, batch_size=batch_size)
    train_ltn(hs_dataloader_train, hs_dataloader_test, args, ndim)


if __name__ == "__main__":
    parser = get_parser()
    generation_args, _ = parser.parse_known_args()
    # We'll also add some additional args for evaluation
    parser.add_argument("--nr_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--log_tensorboard", action='store_true')
    parser.add_argument("--train_data_path", type=str, default='training_dataset_24_08_23.txt')
    parser.add_argument("--test_data_path", type=str, default='test_dataset_24_08_23.txt')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    args = parser.parse_args()
    main(args, generation_args)
