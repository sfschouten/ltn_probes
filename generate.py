from utils import get_parser, load_model, get_dataset, get_dataloader, get_all_hidden_states, save_generations

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    _, dataset_train,_,dataset_test = get_dataset(tokenizer)
    dataloader_train = get_dataloader(dataset_train, tokenizer, batch_size=args.batch_size,
                                num_examples=args.num_examples, device=args.device)
    dataloader_test = get_dataloader(dataset_test, tokenizer, batch_size=args.batch_size,
                                      num_examples=args.num_examples, device=args.device)

    # Get the hidden states and labels
    print("Generating hidden states_train")
    hs = get_all_hidden_states(model, dataloader_train, layer=args.layer, all_layers=args.all_layers)

    # Save the hidden states and labels
    print("Saving hidden states train")
    args.model_name =args.model_name.replace("/","_")+"_train"
    save_generations(hs, args, generation_type="hidden_states")

    print("Generating hidden states test")
    hs = get_all_hidden_states(model, dataloader_test, layer=args.layer, all_layers=args.all_layers)

    # Save the hidden states and labels
    print("Saving hidden states test")
    args.model_name = args.model_name.replace("/", "_").replace("_train","_test")
    save_generations(hs, args, generation_type="hidden_states")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)