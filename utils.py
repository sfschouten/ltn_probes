import os
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM

from auto_gptq import AutoGPTQForCausalLM

from datasets import Dataset


def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="TheBloke/open-llama-7b-open-instruct-GPTQ",
                        help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--data_path", type=str, default='final_city_version_2_train.txt')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    # which hidden states we extract
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states",
                        help="Directory to save the hidden states")

    return parser


def load_model(model_name, cache_dir=None, parallelize=False, device="cuda", use_auto_gptq=None):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """

    use_auto_gptq = use_auto_gptq or 'gptq' in model_name.lower()

    if use_auto_gptq:
        # load the quantized model
        model = AutoGPTQForCausalLM.from_quantized(model_name, device="cuda:0", use_safetensors=True)
        model_type = "decoder-autogtpq"
    else:
        # use the right automodel, and get the corresponding model type
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
            model_type = "encoder_decoder"
        except:
            try:
                model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
                model_type = "encoder"
            except:
                model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
                model_type = "decoder"

    # specify model_max_length (the max token length) to be 512 to ensure that padding works
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type


def get_dataset(tokenizer, data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    sentences = []
    labels = []
    for f in lines:
        text = f.split(",")[0]
        sentences.append(text)
        labels.append(tuple(int(f1) for f1 in f.rstrip().split(",")[1:]))

    data_dict = {'sentence': sentences, "labels": labels}
    data = Dataset.from_dict(data_dict)
    tokenized = data.map(lambda x: tokenizer(
        x['sentence'],
        truncation=True,
        padding="max_length",
    )).with_format("torch") if tokenizer else None
    return data, tokenized


def get_dataloader(dataset, tokenizer, batch_size=16, device="cuda", pin_memory=True, num_workers=1):
    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    # random_idxs = np.random.permutation(len(dataset))

    # remove examples that would be truncated (since this messes up contrast pairs)
    keep_idxs = []
    # for idx in random_idxs:
    for idx in range(len(dataset)):
        input = dataset['sentence'][int(idx)]
        input_text = input.split(",")[0]
        if len(tokenizer.encode(input_text, truncation=False)) < tokenizer.model_max_length - 2:  # include small margin to be conservative
            keep_idxs.append(int(idx))
    dataset = dataset.remove_columns(list(set(dataset.column_names) & {'sentence', 'token_type_ids'}))

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                            num_workers=num_workers)

    return dataloader


def gen_filename(generation_type, arg_dict, exclude_keys):
    name = generation_type + "__" + "__".join(
        ['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys]) + ".npy"
    return name.replace('/', '|')


def save_generations(generation, args, generation_type):
    """
    Input:
        generation: numpy array (e.g. hidden_states or labels) to save
        args: arguments used to generate the hidden states. This is used for the filename to save to.
        generation_type: one of "negative_hidden_states" or "positive_hidden_states" or "labels"

    Saves the generations to an appropriate directory.
    """
    # construct the filename based on the args
    exclude_keys = ["save_dir", "cache_dir", "device"]
    filename = gen_filename(generation_type, vars(args), exclude_keys)

    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save
    np.save(os.path.join(args.save_dir, filename), generation)


def load_single_generation(args, generation_type="hidden_states", name=None):
    # use the same filename as in save_generations
    exclude_keys = ["save_dir", "cache_dir", "device"]
    name = name or gen_filename(generation_type, args, exclude_keys)
    return np.load(os.path.join(args['save_dir'], name))


############# Hidden States #############
def get_first_mask_loc(mask, shift=False):
    """
    return the location of the first pad token for the given ids, which corresponds to a mask value of 0
    if there are no pad tokens, then return the last location
    """
    # add a 0 to the end of the mask in case there are no pad tokens
    mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)

    if shift:
        mask = mask[..., 1:]

    # get the location of the first pad token; use the fact that torch.argmax() returns the first index in the case of ties
    first_mask_loc = torch.argmax((mask == 0).int(), dim=-1)

    return first_mask_loc


def get_individual_hidden_states(model, batch_ids, layer=None, all_layers=True):
    """
    Given a model and a batch of tokenized examples, returns the hidden states for either
    a specified layer (if layer is a number) or for all layers (if all_layers is True).

    If specify_encoder is True, uses "encoder_hidden_states" instead of "hidden_states"
    This is necessary for getting the encoder hidden states for encoder-decoder models,
    but it is not necessary for encoder-only or decoder-only models.
    """

    # forward pass
    with torch.no_grad():
        batch_ids = {key: value.to(model.device) for key, value in batch_ids.items()}
        output = model(**batch_ids, output_hidden_states=True)

    # get all the corresponding hidden states (which is a tuple of length num_layers)
    if "decoder_hidden_states" in output.keys():
        hs_tuple = output["decoder_hidden_states"]
    else:
        hs_tuple = output["hidden_states"]

    # just get the corresponding layer hidden states
    if all_layers:
        # stack along the last axis so that it's easier to consistently index the first two axes
        hs = torch.stack([h.squeeze().detach().cpu() for h in hs_tuple], axis=-1)  # (bs, seq_len, dim, num_layers)
    else:
        assert layer is not None
        hs = hs_tuple[layer].unsqueeze(-1).detach().cpu()  # (bs, seq_len, dim, 1)

    mask = batch_ids["attention_mask"]
    hs[mask == 0] = -torch.inf
    return hs


def get_all_hidden_states(model, dataloader, layer=None, all_layers=True):
    """
    Given a model, a tokenizer, and a dataloader, returns the hidden states (corresponding to a given position index) in all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples *with a candidate label already added* to each example.
    E.g. this function should be used for "Q: Is 2+2=5? A: True" or "Q: Is 2+2=5? A: False", but NOT for "Q: Is 2+2=5? A: ".
    """
    all_hs = []

    model.eval()
    for batch in tqdm(dataloader):
        hs = get_individual_hidden_states(model, batch, layer=layer, all_layers=all_layers)

        if dataloader.batch_size == 1:
            hs = hs.unsqueeze(0)

        all_hs.append(hs)

    all_hs = np.concatenate(all_hs, axis=0)

    return all_hs
