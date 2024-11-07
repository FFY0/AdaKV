#!/bin/bash

MODEL_NAME="Mistral-7B-Instruct-v0.2"
MODEL=${MODELS_DIR}/${MODEL_NAME}
DATASET=${DATASETS_DIR}/LongBench
MAX_LEN=31500

echo "Testing $MODEL $DATASET $MAX_LEN"

scopes=(128 256 512 1024)
device="4090"

for scope in ${scopes[@]}; do
  # ada-snapkv with gqa support
  # we suggest setting floor_alpha to 0.2 while using gqa
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode ada --compress_args_path c"$scope"_w32_k7_maxpool.json --floor_alpha 0.2 --out_name "$device"_ada_"$scope"_"$MODEL_NAME" --gqa_support 
  # snapkv with gqa support
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode fix --compress_args_path c"$scope"_w32_k7_maxpool.json --out_name "$device"_fix_"$scope"_"$MODEL_NAME"  --gqa_support
  # ada-snapkv without gqa support
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode ada --compress_args_path c"$scope"_w32_k7_maxpool.json --floor_alpha 0.2 --out_name "$device"_ada_"$scope"_"$MODEL_NAME"  
  # snapkv without gqa support
  python pred.py -m $MODEL --max_length $MAX_LEN -d $DATASET --mode fix --compress_args_path c"$scope"_w32_k7_maxpool.json --out_name "$device"_fix_"$scope"_"$MODEL_NAME"
done
