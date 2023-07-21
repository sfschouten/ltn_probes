import math

from tqdm import tqdm
import torch
import torch.nn as nn
import ltn
from customdataloader import CustomDataset
from utils import get_parser, load_single_generation, get_dataset, get_dataloader
from transformers import AutoTokenizer

from torch.utils.data import TensorDataset, DataLoader
import neptune
from torch.utils.tensorboard import SummaryWriter
import os

#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SPO_Attention(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)

        self.n_embd = n_embd

        #one = torch.ones(1, 1, n_embd)

        s = torch.ones(1, 1, n_embd)  # torch.randn_like(one)  # TODO these might need a random initialization
        p = torch.ones(1, 1, n_embd)  # torch.randn_like(one)
        o = torch.ones(1, 1, n_embd)  # torch.randn_like(one)
        self.q = nn.Parameter(torch.cat((s, p, o), dim=1))
        # self.q = nn.Parameter(s)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)

        k, v = self.c_attn(x).split(self.n_embd, dim=2)

        mask = mask.all(dim=-1).unsqueeze(1).expand(-1, 1, -1)
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

    def __init__(self, layer_sizes=(4096, 4)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

        layers = [torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))]

        #[torch.nn.init.xavier_uniform(f.weight) for f in layers]


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
        if l!=None:
            out = torch.sum(probs * l, dim=1)
        else:
            out = probs
        return out


def train_ltn(dataloader, dataloader_test, args, ndim):

    run = neptune.init_run(
        project="frankissimo/ltnprobing",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTY1M2U5Ni05ZTU0LTQ0YjAtYWM0OC1jNzUyZTIwOWNiNDQifQ==",

    )  # your credentials

    params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
              "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device}
    run["parameters"] = params


    # Create a summary writer
    writer = SummaryWriter()

    attn = SPO_Attention(ndim)
    attn = attn.to(args.probe_device)

    mlp = MLP(layer_sizes=(4096, 3))
    mlp_sentence = MLP(layer_sizes=(4096*3, 1))
    # mlp2 = MLP()
    # mlp3 = MLP()
    Model = ltn.Predicate(LogitsToPredicate(mlp))
    Model_sentence = ltn.Predicate(LogitsToPredicate(mlp_sentence))
    # Action_model = ltn.Predicate(LogitsToPredicate(mlp))
    # Object_model = ltn.Predicate(LogitsToPredicate(mlp))
    # to cuda
    Model = Model.to(args.probe_device)
    Model_sentence = Model_sentence.to(args.probe_device)
    # Action_model = Action_model.to(args.probe_device)
    # Object_model = Object_model.to(args.probe_device)

    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()
    parameters = []
    parameters.extend([f for f in Model.parameters()])
    parameters.extend([f for f in Model_sentence.parameters()])
    # parameters.extend([f for f in Action_model.parameters()])
    # parameters.extend([f for f in Object_model.parameters()])
    parameters.extend([f for f in attn.parameters()])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    step = 0
    for epoch in tqdm(range(args.nr_epochs)):

        for batch, labels in dataloader:
            step += 1
            optimizer.zero_grad()
            hidden_states = batch
            # hidden_states, = batch

            hidden_states = hidden_states.to("cuda")
            x = hidden_states
            #mask = torch.isfinite(x)
            #x = torch.where(mask, x, 0)
            #x_att = x.sum(dim=1) / mask.all(dim=-1).sum(dim=-1).reshape(-1,1)
            spo = attn(hidden_states)
            #x = ltn.Variable("x", x_att[:, :])
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])
            #y = ltn.Variable("y", x_att[:, :])
            #z = ltn.Variable("z", x_att[:, :])
            Subject_l = ltn.Constant(torch.tensor([1, 0, 0]))
            Action_l = ltn.Constant(torch.tensor([0, 1, 0]))
            Object_l = ltn.Constant(torch.tensor([0, 0, 1]))
            All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))
            label_a = ltn.Variable("label_a", torch.tensor(labels[0].clone().detach()))
            label_b = ltn.Variable("label_b", torch.tensor(labels[1].clone().detach()))
            label_c = ltn.Variable("label_c", torch.tensor(labels[2].clone().detach()))
            label_sentence = ltn.Variable("label_d", torch.tensor(labels[3].clone().detach()))

            subject_positive = Forall(ltn.diag(x, label_a), Model(x, Subject_l),
                                      cond_vars=[label_a],
                                      cond_fn=lambda t: t.value == 1)



            subject_negative = Forall(ltn.diag(x, label_a), Not(Model(x, Subject_l)),
                                      cond_vars=[label_a],
                                      cond_fn=lambda y: y.value == 0)


            action_positive = Forall(ltn.diag(y, label_b), Model(y, Action_l),
                                     cond_vars=[label_b],
                                     cond_fn=lambda t: t.value == 1
                                     )


            action_negative = Forall(ltn.diag(y, label_b), Not(Model(y, Action_l)),
                                     cond_vars=[label_b],
                                     cond_fn=lambda y: y.value == 0
                                     )


            object_positive = Forall(ltn.diag(z, label_c), Model(z, Object_l),
                                     cond_vars=[label_c],
                                     cond_fn=lambda t: t.value == 1
                                     )



            object_negative = Forall(ltn.diag(z, label_c), Not(Model(z, Object_l)),
                                     cond_vars=[label_c],
                                     cond_fn=lambda y: y.value == 0
                                     )

            """
            sentence_score = ltn.Variable("sentence_score", torch.stack(
                (Model(x, Subject_l).value, Model(y, Action_l).value, Model(z, Object_l).value), dim=1))
            """
            """
            sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value),dim=1))

            

            all_sentence_positive = Forall(ltn.diag(x, y, z, label_a, label_b, label_c, label_sentence, sentence_score),
                                           Model_sentence(sentence_score),
                                           cond_vars=[label_sentence],
                                           cond_fn=lambda t: t.value == 1
                                           )

            all_sentence_negative = Forall(ltn.diag(x, y, z, label_a, label_b, label_c, label_sentence, sentence_score),
                                           Not(Model_sentence(sentence_score)),
                                           cond_vars=[label_sentence],
                                           cond_fn=lambda t: t.value == 0
                                           )
            """


            """
            all_sentence_positive_implication = Forall(ltn.diag(x, y, z, label_a, label_b, label_c, sentence_score),
                                                       Implies(Model(x, All_sentence_l),
                                                               And(And(Model(y, Subject_l), Model(z, Action_l)),
                                                                   Model_sentence(sentence_score)),
                                                               cond_vars=[label_sentence],
                                                               cond_fn=lambda y: y.value == 1
                                                                 ))
            
            sat_agg = SatAgg(subject_negative, subject_positive, action_negative, action_positive, object_negative,
                             object_positive, all_sentence_positive_implication, all_sentence_positive,
                             all_sentence_negative)
            
            """
            sat_agg = SatAgg( subject_positive, action_positive, object_positive,
                              subject_negative,action_negative,object_negative)


            #loss = -torch.log(sat_agg)#1. - sat_agg
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
            loss = 1- sat_agg
            # - torch.log(all_sentence_positive_implication.value) - torch.log(all_sentence_positive.value) \
            # - torch.log(all_sentence_negative.value)

            loss.backward()

            """

            # Add the gradient values to Tensorboard
            for name, param in attn.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=step)
            for name, param in Model.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=step)
            for name, param in Model_sentence.named_parameters():
                writer.add_histogram(name + '/grad', param.grad, global_step=step)
            """

            optimizer.step()

            run["train/subject_satisability_positive"].append(subject_positive.value)
            run["train/Action_satisability_positive"].append(action_positive.value)
            run["train/Object_satisability_positive"].append(object_positive.value)
            run["train/subject_satisability_negative"].append(subject_negative.value)
            run["train/Action_satisability_negative"].append(action_negative.value)
            run["train/Object_satisability_negative"].append(object_negative.value)
            #run["train/all_sentence_positive"].append(all_sentence_positive.value)
            #run["train/all_sentence_negative"].append(all_sentence_negative.value)
            #run["train/all_sentence_positive_implication"].append(all_sentence_positive_implication.value)
            run["train/loss"].append(loss)


    Model.eval()
    Model_sentence.eval()
    # Action_model.eval()
    # Object_model.eval()
    attn.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            hidden_states, labels = batch
            hidden_states = hidden_states.to("cuda")
            x = hidden_states
            #mask = torch.isfinite(x)
            #x = torch.where(mask, x, 0)
            #x_att = x.sum(dim=1) / mask.all(dim=-1).sum(dim=-1).reshape(-1, 1)

            spo = attn(hidden_states)
            #x = ltn.Variable("x", x_att[:, :])
            #y = ltn.Variable("y", x_att[:, :])
            #z = ltn.Variable("z", x_att[:, :])
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])
            Subject_l = ltn.Constant(torch.tensor([1, 0, 0]))
            Action_l = ltn.Constant(torch.tensor([0, 1, 0]))
            Object_l = ltn.Constant(torch.tensor([0, 0, 1]))

            All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))
            label_a = ltn.Variable("label_a", torch.tensor(labels[0]))
            label_b = ltn.Variable("label_b", torch.tensor(labels[1]))
            label_c = ltn.Variable("label_c", torch.tensor(labels[2]))
            label_sentence = ltn.Variable("label_d", torch.tensor(labels[3]))

            subject_positive = Forall(ltn.diag(x, label_a), Model(x, Subject_l),
                                      cond_vars=[label_a],
                                      cond_fn=lambda y: y.value == 1)

            subject_negative = Forall(ltn.diag(x, label_a), Not(Model(x, Subject_l)),
                                      cond_vars=[label_a],
                                      cond_fn=lambda y: y.value == 0)

            action_positive = Forall(ltn.diag(y, label_b),
                                     Model(y, Action_l),
                                     cond_vars=[label_b],
                                     cond_fn=lambda y: y.value == 1
                                     )

            action_negative = Forall(ltn.diag(y, label_b), Not(Model(y, Action_l)),
                                     cond_vars=[label_b],
                                     cond_fn=lambda y: y.value == 0
                                     )

            object_positive = Forall(ltn.diag(z, label_c),
                                     Model(z, Object_l),
                                     cond_vars=[label_c],
                                     cond_fn=lambda y: y.value == 1
                                     )

            object_negative = Forall(ltn.diag(z, label_c), Not(Model(z, Object_l)),
                                     cond_vars=[label_c],
                                     cond_fn=lambda y: y.value == 0
                                     )
            """
            sentence_score = ltn.Variable("sentence_score", torch.stack(
                (Model(x, Subject_l).value, Model(y, Action_l).value, Model(z, Object_l).value), dim=1))
            """

            sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value),dim=1))
            """


            all_sentence_positive = Forall(ltn.diag(x, y, z, label_a, label_b, label_c,
                                                    label_sentence, sentence_score),
                                           Model_sentence(sentence_score),
                                           cond_vars=[label_sentence],
                                           cond_fn=lambda y: y.value == 1
                                           )

            all_sentence_negative = Forall(ltn.diag(x, y, z, label_a, label_b, label_c,
                                                    label_sentence, sentence_score),
                                           Not(Model_sentence(sentence_score)),
                                           cond_vars=[label_sentence],
                                           cond_fn=lambda y: y.value == 0
                                           )


            

            all_sentence_positive_implication = Forall(ltn.diag(x, y, z, label_a, label_b, label_c, sentence_score),
                                                       Implies(Model(x, All_sentence_l),
                                                               And(And(Model(y, Subject_l), Model(z, Action_l)),
                                                                   Model_sentence(sentence_score)),
                                                               cond_vars=[label_sentence],
                                                               cond_fn=lambda y: y.value == 1
                                                               ))
            """
            file = open("city_test_cleaned.txt", "r")
            for iteration in file:
                print(iteration)

            print("PETER IN SENTENCE", Model(x, Subject_l).value)
            print("(LIVE) IN SENTENCE", Model(x, Action_l).value)
            print("AMSTERDAM IN SENTENCE", Model(x, Object_l).value)
            print("AMSTERDAM LIVES IN Peter SENTENCE", Model_sentence(sentence_score).value)



            run["test/subject_satisability_positive"].append(subject_positive.value)
            run["test/Action_satisability_positive"].append(action_positive.value)
            run["test/Object_satisability_positive"].append(object_positive.value)
            run["test/subject_satisability_negative"].append(subject_negative.value)
            run["test/Action_satisability_negative"].append(action_negative.value)
            run["test/Object_satisability_negative"].append(object_negative.value)
            #run["test/all_sentence_positive"].append(all_sentence_positive.value)
            #run["test/all_sentence_negative"].append(all_sentence_negative.value)
            #run["test/all_sentence_positive_implication"].append(all_sentence_positive_implication.value)
            run["test/loss"].append(1. - sat_agg)


            # print(loss)

    run.stop()


def main(args, generation_args):
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    # load dataset and hidden states
    hs_train = load_single_generation(generation_args,
                                      name="hidden_states__model_name_TheBloke_open-llama-7b-open-instruct-GPTQ_train__parallelize_False__batch_size_1__num_examples_1000__layer_-1__all_layers_False.npy")
    hs_test = load_single_generation(generation_args,
                                     name="hidden_states__model_name_TheBloke_open-llama-7b-open-instruct-GPTQ_test__parallelize_False__batch_size_1__num_examples_1000__layer_-1__all_layers_False.npy")

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
    hs_dataset_train = CustomDataset(hs_train_t, dataset_train)
    batch_size = args.probe_batch_size if args.probe_batch_size > 0 else len(hs_train_t)
    hs_dataloader_train = DataLoader(hs_dataset_train, batch_size=batch_size, shuffle=True)

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
    parser.add_argument("--nr_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=128)
    parser.add_argument("--probe_device", type=str, default='cuda')
    args = parser.parse_args()
    main(args, generation_args)
