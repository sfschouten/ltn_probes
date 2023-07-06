from utils import get_parser, load_model, get_dataset, get_dataloader, get_all_hidden_states, save_generations

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    _, dataset = get_dataset(tokenizer)
    dataloader = get_dataloader(dataset, tokenizer, batch_size=args.batch_size, 
                                num_examples=args.num_examples, device=args.device)

    # Get the hidden states and labels
    print("Generating hidden states")
    hs = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers)

    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(hs, args, generation_type="hidden_states")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
