# # # # # # # # #
#   OpenLLaMA   #
# # # # # # # # #
# forward model on data, generating hidden states
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=0
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=4
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=8
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=12
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=16
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=20
python generate.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=24

# baseline: train probe with labels shuffled w.r.t. features
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=0 --log_neptune --write_metrics --metrics_file=metrics_bert_0_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=4 --log_neptune --write_metrics --metrics_file=metrics_bert_4_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=8 --log_neptune --write_metrics --metrics_file=metrics_bert_8_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=12 --log_neptune --write_metrics --metrics_file=metrics_bert_12_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=16 --log_neptune --write_metrics --metrics_file=metrics_bert_16_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=20 --log_neptune --write_metrics --metrics_file=metrics_bert_20_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=24 --log_neptune --write_metrics --metrics_file=metrics_bert_24_shuffled.csv --shuffled_baseline

# probing
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=0 --log_neptune --write_metrics --metrics_file=metrics_bert_0.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=4 --log_neptune --write_metrics --metrics_file=metrics_bert_4.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=8 --log_neptune --write_metrics --metrics_file=metrics_bert_8.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=12 --log_neptune --write_metrics --metrics_file=metrics_bert_12.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=16 --log_neptune --write_metrics --metrics_file=metrics_bert_16.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=20 --log_neptune --write_metrics --metrics_file=metrics_bert_20.csv
python evaluate_fn.py --model=TheBloke/open-llama-7b-open-instruct-GPTQ --dataset=framenet --layer=24 --log_neptune --write_metrics --metrics_file=metrics_bert_24.csv



# # # # # # # # #
#      BERT     #
# # # # # # # # #
# forward model on data, generating hidden states
python generate.py --model=bert-large-uncased --dataset=framenet --layer=0
python generate.py --model=bert-large-uncased --dataset=framenet --layer=4
python generate.py --model=bert-large-uncased --dataset=framenet --layer=8
python generate.py --model=bert-large-uncased --dataset=framenet --layer=12
python generate.py --model=bert-large-uncased --dataset=framenet --layer=16
python generate.py --model=bert-large-uncased --dataset=framenet --layer=20
python generate.py --model=bert-large-uncased --dataset=framenet --layer=24

# baseline: train probe with labels shuffled w.r.t. features
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=0 --log_neptune --write_metrics --metrics_file=metrics_bert_0_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=4 --log_neptune --write_metrics --metrics_file=metrics_bert_4_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=8 --log_neptune --write_metrics --metrics_file=metrics_bert_8_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=12 --log_neptune --write_metrics --metrics_file=metrics_bert_12_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=16 --log_neptune --write_metrics --metrics_file=metrics_bert_16_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=20 --log_neptune --write_metrics --metrics_file=metrics_bert_20_shuffled.csv --shuffled_baseline
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=24 --log_neptune --write_metrics --metrics_file=metrics_bert_24_shuffled.csv --shuffled_baseline

# probing
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=0 --log_neptune --write_metrics --metrics_file=metrics_bert_0.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=4 --log_neptune --write_metrics --metrics_file=metrics_bert_4.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=8 --log_neptune --write_metrics --metrics_file=metrics_bert_8.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=12 --log_neptune --write_metrics --metrics_file=metrics_bert_12.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=16 --log_neptune --write_metrics --metrics_file=metrics_bert_16.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=20 --log_neptune --write_metrics --metrics_file=metrics_bert_20.csv
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=24 --log_neptune --write_metrics --metrics_file=metrics_bert_24.csv



# baseline: random features (model/layer doesn't matter)
python evaluate_fn.py --model=bert-large-uncased --dataset=framenet --layer=0 --log_neptune --write_metrics --metrics_file=metrics_bert_random.csv --random_baseline
