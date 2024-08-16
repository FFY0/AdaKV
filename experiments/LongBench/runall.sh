#!/bin/bash

scopes=(1024 512 256 128)

# Test Llama
#
# MODEL_NAME="LWM-Text-Chat-1M"
# MODEL=${MODELS_DIR}/${MODEL_NAME}
# DATASET=${DATASETS_DIR}/LongBench
# MAX_LEN=1048576
# MODEL_NAME="tiny-mistral"

MODEL_NAME="Mistral-7B-Instruct-v0.2"
MODEL=${MODELS_DIR}/${MODEL_NAME}
DATASET=${DATASETS_DIR}/LongBench
MAX_LEN=31500

echo "Testing $MODEL $DATASET $MAX_LEN"

for scope in ${scopes[@]}; do
  # ada model
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode ada --compress_args_path c"$scope"_w32_k7_maxpool.json --floor_alpha 0.5 --out_name ada_"$scope"_"$MODEL_NAME" 
  # snap model
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode fix --compress_args_path c"$scope"_w32_k7_maxpool.json --out_name fix_"$scope"_"$MODEL_NAME"
  # ada pyram model
  # python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode ada --compress_args_path c"$scope"_w32_k7_maxpool.json --floor_alpha 0.5 --pyram --out_name ada_pyram_"$scope"_"$MODEL_NAME"
  # snap pyram model
  # python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode fix --compress_args_path c"$scope"_w32_k7_maxpool.json --pyram --out_name fix_pyram_"$scope"_"$MODEL_NAME"
done

# base model
python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --compress_args_path c"$scope"_w32_k7_maxpool.json --out_name base_"$MODEL_NAME"


