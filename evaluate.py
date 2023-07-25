import math
import os

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader

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
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)

        self.n_embd = n_embd

        s = torch.randn(1, 1, n_embd)
        p = torch.randn(1, 1, n_embd)
        o = torch.randn(1, 1, n_embd)
        self.q = nn.Parameter(torch.cat((s, p, o), dim=1))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, return_attn_values=True):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)

        k, v = self.c_attn(x).split(self.n_embd, dim=2)

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
        return out


def create_axioms(Model, Model_sentence, Subject_l, Action_l, Object_l, labels, x, y, z, sentence_score):
    # we define the connectives, quantifiers, and the SatAgg
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    label_a = ltn.Variable("label_a", labels[0].clone().detach())
    label_b = ltn.Variable("label_b", labels[1].clone().detach())
    label_c = ltn.Variable("label_c", labels[2].clone().detach())
    label_sentence = ltn.Variable("label_d", labels[3].clone().detach())

    subject_positive = Forall(ltn.diag(x, label_a), Model(x, Subject_l),
                              cond_vars=[label_a],
                              cond_fn=lambda t: t.value == 1)

    subject_negative = Forall(ltn.diag(x, label_a), Not(Model(x, Subject_l)),
                              cond_vars=[label_a],
                              cond_fn=lambda t: t.value == 0)

    action_positive = Forall(ltn.diag(y, label_b), Model(y, Action_l),
                             cond_vars=[label_b],
                             cond_fn=lambda t: t.value == 1
                             )

    action_negative = Forall(ltn.diag(y, label_b), Not(Model(y, Action_l)),
                             cond_vars=[label_b],
                             cond_fn=lambda t: t.value == 0
                             )

    object_positive = Forall(ltn.diag(z, label_c), Model(z, Object_l),
                             cond_vars=[label_c],
                             cond_fn=lambda t: t.value == 1
                             )

    object_negative = Forall(ltn.diag(z, label_c), Not(Model(z, Object_l)),
                             cond_vars=[label_c],
                             cond_fn=lambda t: t.value == 0
                             )
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

    sat_agg = SatAgg(subject_negative, subject_positive, action_negative, action_positive, object_negative,
                     object_positive, all_sentence_positive_implication, all_sentence_positive,
                     all_sentence_negative, all_sentence_negative_implication)

    """
    sat_agg = SatAgg(subject_positive, action_positive, object_positive,
                     subject_negative, action_negative, object_negative,all_sentence_positive,all_sentence_negative)
    """
    return {
        'subject_positive': subject_positive,
        'subject_negative': subject_negative,
        'action_positive': action_positive,
        'action_negative': action_negative,
        'object_positive': object_positive,
        'object_negative': object_negative,
        'all_sentence_positive': all_sentence_positive,
        'all_sentence_negative': all_sentence_negative,
        'all_sentence_positive_implication': all_sentence_positive_implication,
        'all_sentence_negative_implication': all_sentence_negative_implication,
    }, sat_agg


def train_ltn(dataloader, dataloader_test, args, ndim):
    if args.log_neptune:
        import neptune
        run = neptune.init_run(
            project=os.getenv('NEPTUNE_PROJECT'),
            api_token=os.getenv('NEPTUNE_API_KEY'),
        )  # your credentials

        params = {"learning_rate": args.lr, "optimizer": "Adam", "nr_epochs": args.nr_epochs,
                  "probe_batch_size": args.probe_batch_size, "probe_device": args.probe_device}
        run["parameters"] = params

    if args.log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Create a summary writer
        writer = SummaryWriter()

    #num_heads = 3
    #heads_per_dim = 4096
    #embed_dimension = 4096  # num_heads * heads_per_dim
    # input_dim, embed_dim, num_heads
    # attn = ScaledDotProductAttentionModel(4096,3)
    # attn = attn.to(args.probe_device)

    attn = SPOAttention(ndim)
    attn = attn.to(args.probe_device)

    mlp = MLP(layer_sizes=(ndim, 3))
    mlp_sentence = MLP(layer_sizes=(ndim*3, 1))
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

    Subject_l = ltn.Constant(torch.tensor([1, 0, 0]))
    Action_l = ltn.Constant(torch.tensor([0, 1, 0]))
    Object_l = ltn.Constant(torch.tensor([0, 0, 1]))
    All_sentence_l = ltn.Constant(torch.tensor([0, 0, 0, 1]))

    parameters = []
    parameters.extend([f for f in Model.parameters()])
    parameters.extend([f for f in Model_sentence.parameters()])
    # parameters.extend([f for f in Action_model.parameters()])
    # parameters.extend([f for f in Object_model.parameters()])
    parameters.extend([f for f in attn.parameters()])
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    step = 0
    for _ in tqdm(range(args.nr_epochs)):
        for hs, labels in dataloader:
            hs = hs.to(args.probe_device)

            # forward attention
            spo, _ = attn(hs)
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])

            # sentence_score = ltn.Variable("sentence_score", torch.stack( (Model(x, Subject_l).value, Model(y, Action_l).value, Model(z, Object_l).value), dim=1))
            sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value), dim=1))

            # create axioms
            axioms, sat_agg = create_axioms(Model, Model_sentence, Subject_l, Action_l, Object_l, labels, x, y, z, sentence_score)

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
    Model_sentence.eval()
    # Action_model.eval()
    # Object_model.eval()
    attn.eval()
    with torch.no_grad():
        for hs, labels in tqdm(dataloader_test):
            hs = hs.to(args.probe_device)

            spo, attention_values = attn(hs)
            x = ltn.Variable("x", spo[:, 0, :])
            y = ltn.Variable("y", spo[:, 1, :])
            z = ltn.Variable("z", spo[:, 2, :])

            sentence_score = ltn.Variable("sentence_score", torch.concat((x.value, y.value, z.value), dim=1))

            axioms, _ = create_axioms(Model, Model_sentence, Subject_l, Action_l, Object_l, labels, x, y, z, sentence_score)

            torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)

            with open("city_test_cleaned.txt", "r") as file:
                for iteration in file:
                    print(iteration)

            print(f"PETER = SUBJECT                       {Model(x, Subject_l).value}")
            print(f"(LIVE) IN SENTENCE                    {Model(y, Action_l).value}")
            print(f"AMSTERDAM = OBJECT                    {Model(z, Object_l).value}")
            print(f"'PETER LIVES IN AMSTERDAM' = SENTENCE {Model_sentence(sentence_score).value}")

            print(attention_values)

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

    dataset_train, _ = get_dataset(None, args.train_data_path)
    dataset_test, _ = get_dataset(None, args.test_data_path)

    def trim_hidden_states(hs):
        mask = np.isfinite(hs)          # (nr_samples, nr_layers, nr_tokens, nr_dims)
        mask = mask.all(axis=3)
        token_cnt = mask.sum(axis=2)
        trim_i = token_cnt.max()
        print(f'trimming to {trim_i} from {hs.shape[2]}')
        return hs[:, :, :trim_i, :]

    # load dataset and hidden states
    if not args.random_baseline:
        hs_train = trim_hidden_states(load_single_generation(vars(generation_args) | {'data_path': args.train_data_path}))
        hs_test = trim_hidden_states(load_single_generation(vars(generation_args) | {'data_path': args.test_data_path}))
    else:
        hs_train = np.random.randn(len(dataset_train), 32, 512)
        hs_test = np.random.randn(len(dataset_test), 32, 512)

    # train LTN probe
    hs_train_t = torch.Tensor(hs_train).squeeze()
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
    parser.add_argument("--nr_epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--log_neptune", action='store_true')
    parser.add_argument("--log_tensorboard", action='store_true')
    parser.add_argument("--train_data_path", type=str, default='final_city_version_2_train.txt')
    parser.add_argument("--test_data_path", type=str, default='city_test_cleaned.txt')
    parser.add_argument("--random_baseline", action='store_true',
                        help="Use randomly generated 'hidden states' to test if probe is able to learn with a random "
                             "set of numbers, indicating that we are not really probing.")
    args = parser.parse_args()
    main(args, generation_args)
