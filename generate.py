from utils import get_parser, load_model, get_dataloader, get_all_hidden_states, save_generations, \
                    get_synthetic_dataset, get_framenet_dataset


def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, _ = load_model(args.model_name, cache_dir=args.cache_dir, device=args.device)

    print("Loading dataloader")
    if args.dataset == 'synthetic':
        _, dataset = get_synthetic_dataset(tokenizer, args.data_path)
        dataset = dataset.remove_columns(['labels'])
    elif args.dataset == 'framenet':
        dataset, _, _, _, _ = get_framenet_dataset(tokenizer)
        dataset = dataset.remove_columns(['frames', 'labels'])
    dataloader = get_dataloader(dataset, tokenizer, batch_size=args.batch_size)

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
