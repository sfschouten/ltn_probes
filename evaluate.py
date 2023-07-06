from utils import get_parser, load_single_generation, get_dataset
from transformers import AutoTokenizer
import torch.nn as nn

class SPO_Attention(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        
        self.n_embd = n_embd

        s = torch.ones(1, 1, n_embd) # TODO these might need a random initialization
        p = torch.ones(1, 1, n_embd)
        o = torch.ones(1, 1, n_embd)
        self.q = nn.Parameter(torch.cat((s,p,o), dim=1))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        # calculate key, values
        mask = torch.isfinite(x)
        x = torch.where(mask, x, 0)
        k, v = self.c_attn(x).split(self.n_embd, dim=2)

        mask = mask.all(dim=-1).unsqueeze(1).expand(-1, self.n_queries, -1)
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



def train_ltn(dataloader, args):


    #attn = SPO_Attention(args.probe_ndim)

    #for epoch in range(args.nr_epochs):
        
    #    for batch in dataloader:
           
    #        spo = attn(batch)


    # ...
    pass


def main(args, generation_args):
    # load dataset and hidden states 
    hs = load_single_generation(generation_args)

    tokenizer = AutoTokenizer.from_pretrained(generation_args.model_name, use_fast=True)
    dataset, tokenized = get_dataset(tokenizer)

    for hs_row, sample in zip(hs, dataset):
        
        print(hs_row)
        #print(sample)

        # sample_enc to enable recovering which token belonged to which word, in case we need it.
        sample_enc = tokenizer(sample)

    # train LTN probe
    dataloader = get_dataloader(dataset, tokenized, batch_size=args.probe_batch_size, device=args.probe_device)
    train_ltn(dataloader, args)


if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args() 
    # We'll also add some additional args for evaluation
    parser.add_argument("--nr_epochs", type=int, default=1000)    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=-1)
    parser.add_argument("--probe_device", type=str, default='cuda')
    parser.add_argument("--probe_ndim", type=int, default=128)
    args = parser.parse_args()
    main(args, generation_args)
