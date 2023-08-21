import math
import os

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ltn

from utils import get_parser, load_single_generation, get_synthetic_dataset
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

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.MatrixVector = nn.Parameter(2e-4 * torch.rand([4096, 2099]), requires_grad=True)

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


def create_axioms(Model, Model_person, Subject_l, Action_l, Object_l, labels, x, y, z, sentence_score=None):
    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=4.0))

    # label_a = ltn.Variable("label_a", labels[0].clone().detach())
    # label_b = ltn.Variable("label_b", labels[1].clone().detach())
    # label_c = ltn.Variable("label_c", labels[2].clone().detach())
    label_sentence = ltn.Variable("label_d", labels[3].clone().detach())  # sentence correctly
    label_person_subject = ltn.Variable("label_person_subject", labels[4].clone().detach())
    label_person_action = ltn.Variable("label_person_action", labels[5].clone().detach())
    label_person_object = ltn.Variable("label_person_object", labels[6].clone().detach())

    """
    for v in x.value:
        if not torch.all(torch.where(torch.logical_and(v >= 0., v <= 1.), 1., 0.)):
            print("dddd")
    """
    # print( Model(x, Subject_l).value)
    """
    subject_positive = Forall(ltn.diag(x, Subject_l), Model(x, Subject_l)
    action_positive = Forall(ltn.diag(y, Action_l), Model(y, Action_l))
    object_positive = Forall(ltn.diag(z, Object_l), Model(z, Object_l))
    """

    subject_positive = Forall(ltn.diag(label_person_subject, x, Subject_l),
                              And(Model(x, Subject_l), Model_person(x)),
                              cond_vars=[label_person_subject],
                              cond_fn=lambda t: t.value == 1)

    object_positive = Forall(ltn.diag(label_person_object, z, Object_l),
                             And(Model(z, Object_l), Model_person(z)),
                             cond_vars=[label_person_object],
                             cond_fn=lambda t: t.value == 1)

    action_positive = Forall(ltn.diag(y, Action_l), Model(y, Action_l))

    is_not_person_subject = Forall(ltn.diag(label_person_subject, x, Subject_l),
                                   And(Model(x, Subject_l), Not(Model_person(x))),
                                   cond_vars=[label_person_subject],
                                   cond_fn=lambda t: t.value == 0)

    is_not_person_object = Forall(ltn.diag(label_person_object, z, Object_l),
                                  And(Model(z, Object_l), Not(Model_person(z))),
                                  cond_vars=[label_person_object],
                                  cond_fn=lambda t: t.value == 0)

    """
    is_person_subject = Forall(ltn.diag(label_person_subject, x),
                               Model_person(x),
                               cond_vars=[label_person_subject],
                               cond_fn=lambda t: t.value == 1)

    is_not_person_subject = Forall(ltn.diag(label_person_subject, x),
                                   Not(Model_person(x)),
                                   cond_vars=[label_person_subject],
                                   cond_fn=lambda t: t.value == 0)

    is_person_object = Forall(ltn.diag(label_person_object, z),
                              Model_person(z),
                              cond_vars=[label_person_object],
                              cond_fn=lambda t: t.value == 1)

    is_not_person_object = Forall(ltn.diag(label_person_object, z),
                                  Not(Model_person(z)),
                                  cond_vars=[label_person_object],
                                  cond_fn=lambda t: t.value == 0)
                                  
    """

    """

    subject_negative = Forall(ltn.diag(x, label_a), Not(Model(x, Subject_l)),
                              cond_vars=[label_a],
                              cond_fn=lambda t: t.value == 0)

    action_negative = Forall(ltn.diag(y, label_b), Not(Model(y, Action_l)),
                             cond_vars=[label_b],
                             cond_fn=lambda t: t.value == 0)

    object_negative = Forall(ltn.diag(z, label_c), Not(Model(z, Object_l)),
                             cond_vars=[label_c],
                             cond_fn=lambda t: t.value == 0)

    """

    """
    all_sentence_positive = Forall(ltn.diag(label_sentence, sentence_score),
                                   Model_sentence(sentence_score),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 1)

    all_sentence_negative = Forall(ltn.diag(label_sentence, sentence_score),
                                   Not(Model_sentence(sentence_score)),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 0)
    
    """

    """



    all_sentence_positive = Forall(ltn.diag(label_sentence, sentence_score,x,y,z,Subject_l,Object_l,Action_l),
                                   Implies(And(And(Model(x, Subject_l),Model(y, Action_l)),Model(z, Object_l)),
                                               Model_sentence(sentence_score)),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 1)

    all_sentence_negative = Forall(ltn.diag(label_sentence, sentence_score,x,y,z,Subject_l,Object_l,Action_l),
                                   Implies(And(Model(x, Subject_l),Model(z, Object_l)),
                                               Not(Model_sentence(sentence_score))),
                                   cond_vars=[label_sentence],
                                   cond_fn=lambda t: t.value == 0)
    """

    """

    all_sentence_positive_implication = Forall(
        ltn.diag(x, y, z, label_a, label_b, label_c, sentence_score, label_sentence),
        Implies(Model_sentence(sentence_score),
                And(And(Model(x, Subject_l), Model(y, Action_l)),
                    Model(z, Object_l))),

        cond_vars=[label_sentence],
        cond_fn=lambda t: t.value == 1
    )

    all_sentence_negative_implication = Forall(
        ltn.diag(x, y, z, label_a, label_b, label_c, sentence_score, label_sentence),
        Implies(Not(Model_sentence(sentence_score)),
                Or(Or(Not(Model(x, Subject_l)), Not(Model(y, Action_l))),
                   Not(Model(z, Object_l)))),

        cond_vars=[label_sentence],
        cond_fn=lambda t: t.value == 0
    )
    """""

    sat_agg = SatAgg(subject_positive, action_positive, object_positive, is_not_person_subject,
                     is_not_person_object)

    # sat_agg = SatAgg(subject_positive, action_positive, object_positive,
    #                  subject_negative, action_negative, object_negative, all_sentence_positive, all_sentence_negative)

    return {
        'subject_positive': subject_positive,
        # 'subject_negative': subject_negative,
        'action_positive': action_positive,
        # 'action_negative': action_negative,
        'object_positive': object_positive,
        # 'object_negative': object_negative,
        # 'all_sentence_positive': all_sentence_positive,
        # 'all_sentence_negative': all_sentence_negative,
        # 'is_person_subject': is_person_subject,
        'is_not_person_subject': is_not_person_subject,
        # 'is_person_object': is_person_object,
        'is_not_person_object': is_not_person_object,
        # 'all_sentence_positive_implication': all_sentence_positive_implication,
        # 'all_sentence_negative_implication': all_sentence_negative_implication,
    }, sat_agg


def get_score_ltn(x, y):
    # print("function")
    return torch.gather(x, 1, y.view(-1, 1)).view(-1)


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
    mlp_person = MLP(layer_sizes=(2099, 1))

    # mlp2 = MLP()
    # mlp3 = MLP()

    Model = ltn.Function(func=lambda x, y: get_score_ltn(x, y))
    # Model_sentence = ltn.Predicate(LogitsToPredicate(mlp_sentence)).to(args.probe_device)
    Model_person = ltn.Predicate(LogitsToPredicate(mlp_person)).to(args.probe_device)
    # Action_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Object_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Action_model = Action_model.to(args.probe_device)
    # Object_model = Object_model.to(args.probe_device)

    parameters = []
    # parameters.extend([MatrixVector])
    parameters.extend([f for f in Model_person.parameters()])
    # parameters.extend([f for f in Action_model.parameters()])
    # parameters.extend([f for f in Object_model.parameters()])

    parameters.extend([f for f in attn.parameters()])
    # parameters.extend([f for f in attn.v_proj.parameters()])

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0)

    step = 0
    for _ in tqdm(range(args.nr_epochs)):
        for hs, _, labels in dataloader:
            hs = hs.to(args.probe_device)

            # forward attention
            spo, _ = attn(hs)
            subject = spo[:, 0, :]
            action = spo[:, 1, :]
            object = spo[:, 2, :]
            scores_sub = subject.mm(attn.MatrixVector)
            score_action = action.mm(attn.MatrixVector)
            score_object = object.mm(attn.MatrixVector)

            x = ltn.Variable("x", torch.softmax(scores_sub, dim=1))
            y = ltn.Variable("y", torch.softmax(score_action, dim=1))
            z = ltn.Variable("z", torch.softmax(score_object, dim=1))

            Subject_l = ltn.Variable("x_label", labels[0])
            Action_l = ltn.Variable("y_label", labels[1])
            Object_l = ltn.Variable("z_label", labels[2])
            # All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))

            labels[0] = labels[0].to(args.probe_device)
            labels[1] = labels[1].to(args.probe_device)
            labels[2] = labels[2].to(args.probe_device)

            """
            sentence_score = ltn.Variable("sentence_score", torch.concat((subject,action,object, torch.gather(x.value,1,Subject_l.value),
                                                                           torch.gather(y.value,1,Action_l.value),
                                                                           torch.gather(z.value,1,Object_l.value)), dim=1))
            """
            # sentence_score = ltn.Variable("sentence_score",torch.concat((x.value, y.value, z.value), dim=1))

            # create axioms
            axioms, sat_agg = create_axioms(Model, Model_person, Subject_l, Action_l, Object_l, labels, x, y, z)

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
                    run[f'train/{key}'].append(value.value)
                run["train/loss"].append(loss)

            step += 1

    Model.eval()
    Model_person.eval()
    # Action_model.eval()
    # Object_model.eval()
    attn.eval()

    with torch.no_grad():
        for hs, sentences, labels in tqdm(dataloader_test):
            hs = hs.to(args.probe_device)

            spo, _ = attn(hs)
            subject = spo[:, 0, :]
            action = spo[:, 1, :]
            object = spo[:, 2, :]
            scores_sub = subject.mm(attn.MatrixVector)
            score_action = action.mm(attn.MatrixVector)
            score_object = object.mm(attn.MatrixVector)

            x = ltn.Variable("x", torch.softmax(scores_sub, dim=1))
            y = ltn.Variable("y", torch.softmax(score_action, dim=1))
            z = ltn.Variable("z", torch.softmax(score_object, dim=1))

            Subject_l = ltn.Variable("x_label", labels[0])
            Action_l = ltn.Variable("y_label", labels[1])
            Object_l = ltn.Variable("z_label", labels[2])
            All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))

            labels[0] = labels[0].to(args.probe_device)
            labels[1] = labels[1].to(args.probe_device)
            labels[2] = labels[2].to(args.probe_device)

            ltn.Variable("sentence_score",
                         torch.concat((subject, action, object, torch.gather(x.value, 1, Subject_l.value),
                                       torch.gather(y.value, 1, Action_l.value),
                                       torch.gather(z.value, 1, Object_l.value)), dim=1))

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
            Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6), quantifier="f")

            torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
            print(sentences)
            print(f" SUBJECT                      ", torch.argmax(x.value, dim=1), labels[0])
            print(f"(LIVE) IN SENTENCE                   ", torch.argmax(y.value, dim=1), labels[1])
            print(f" OBJECT                   ", torch.argmax(z.value, dim=1), labels[2])
            print(f" PERSON - subject                      ", Model_person(x), labels[4])
            print(f" PERSON - object                      ", Model_person(z), labels[6])

            label_subject = ltn.Variable("subject_label", torch.argmax(x.value, dim=1))
            label_action = ltn.Variable("subject_action", torch.argmax(y.value, dim=1))
            label_object = ltn.Variable("subject_object", torch.argmax(z.value, dim=1))

            label_sentence = ltn.Variable("label_d", labels[3].clone().detach())  # sentence correctly
            label_person_subject = ltn.Variable("label_person_subject", labels[4].clone().detach())
            label_person_action = ltn.Variable("label_person_action", labels[5].clone().detach())
            label_person_object = ltn.Variable("label_person_object", labels[6].clone().detach())




            subject_max = torch.max(x.value,dim=1)[0]
            action_max = torch.max(y.value, dim=1)[0]
            object_max = torch.max(z.value, dim=1)[0]
            person_max = Model_person(x).value
            object_person = Model_person(z).value




            subject_positive = Forall(ltn.diag( x, label_subject,y,label_action,z,label_object,label_sentence),
                                      And(And(And(Model(x, label_subject), Model_person(x)),
                                              Model(y, label_action)),
                                          And(Model(z, label_object), Not(Model_person(z)))),
                                      cond_vars=[label_sentence],
                                      cond_fn=lambda t: t.value == 1)

            subject_negative = Forall(ltn.diag(x, label_subject, y, label_action, z, label_object, label_sentence),
                                      And(And(And(Model(x, label_subject), Model_person(x)),
                                              Model(y, label_action)),
                                          And(Model(z, label_object), Not(Model_person(z)))),
                                      cond_vars=[label_sentence],
                                      cond_fn=lambda t: t.value == 0)




            print(f" Sentences positive                  ",subject_positive.value)
            print(f" Sentences negative                  ", subject_negative.value)


            print("Sentences positive - for each row",torch.min(torch.min(torch.min(subject_max,person_max),action_max),torch.min(object_max,1-object_person)))
                  

            # print(f"'PETER LIVES IN AMSTERDAM' = SENTENCE", Model_sentence(sentence_score).value, labels[3])
            # print(attention_values)

            if args.log_neptune:
                for key, value in axioms.items():
                    run[f'test/{key}'].append(value.value)

    if args.log_neptune:
        run.stop()


def main(args, generation_args):
    print(f'CUDA available? {torch.cuda.is_available()}')
    print(f'Nr. of CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Name of first device: {torch.cuda.get_device_name(0)}')
    print()

    dataset_train, _ = get_synthetic_dataset(None, args.train_data_path)
    dataset_test, _ = get_synthetic_dataset(None, args.test_data_path)

    def trim_hidden_states(hs):
        mask = np.isfinite(hs)  # (nr_samples, nr_layers, nr_tokens, nr_dims)
        mask = mask.all(axis=3)
        token_cnt = mask.sum(axis=2)
        trim_i = token_cnt.max()
        print(f'Trimming to {trim_i} from {hs.shape[2]}.')
        return hs[:, :, :trim_i, :]

    # load dataset and hidden states
    if not args.random_baseline:
        generation_args.data_path = args.train_data_path
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args)))
        generation_args.data_path = args.test_data_path
        hs_test = trim_hidden_states(load_single_generation(vars(generation_args)))
    else:
        hs_train = np.random.randn(len(dataset_train), 32, 512)
        hs_test = np.random.randn(len(dataset_test), 32, 512)

    # train LTN probe
    hs_train_t = torch.Tensor(hs_train).squeeze()
    hs_dataset_train = CustomDataset(hs_train_t, dataset_train)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_train_t)
    hs_dataloader_train = DataLoader(hs_dataset_train, batch_size=batch_size, shuffle=True)
    val = [f[0] for f in dataset_train["labels"]]
    val.extend([f[1] for f in dataset_train["labels"]])
    val.extend([f[2] for f in dataset_train["labels"]])

    print("tot actions", len(list(set([f[1] for f in dataset_train["labels"]]))))
    print("tot subjects", len(list(set([f[0] for f in dataset_train["labels"]]))))
    print("tot objects", len(list(set([f[2] for f in dataset_train["labels"]]))))

    print("max_elements", max(list(val)))

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
    parser.add_argument("--nr_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--log_tensorboard", action='store_true')
    parser.add_argument("--train_data_path", type=str, default='training_set_26_07_23.txt')
    parser.add_argument("--test_data_path", type=str, default='test_set_26_07_23.txt')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    args = parser.parse_args()
    main(args, generation_args)
