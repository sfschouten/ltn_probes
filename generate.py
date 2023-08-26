from utils import get_parser, load_model, get_dataset, get_dataloader, get_all_hidden_states, save_generations


def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, _ = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    _, dataset = get_dataset(tokenizer,args.data_path)
    dataset = dataset.remove_columns(['labels'])
    dataloader = get_dataloader(dataset, tokenizer, batch_size=args.batch_size, device=args.device)

    # Get the hidden states and labels
    print("Generating hidden states")
    hs = get_all_hidden_states(model, dataloader, layer=args.layer, all_layers=args.all_layers)

    # Save the hidden states and labels
    print("Saving hidden states")
    # args.model_name = args.model_name.replace("/", "_")+"_train"
    save_generations(hs, args, generation_type="hidden_states")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
