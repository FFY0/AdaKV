
conda activate snapekv
#CUDA_VISIBLE_DEVICES=0,1 python ./longbench_eval.py  &
CUDA_VISIBLE_DEVICES=1 python ./longbench_eval.py --compress_args_path c256_w32_k7_maxpool.json &
CUDA_VISIBLE_DEVICES=2 python ./longbench_eval.py --compress_args_path c512_w32_k7_maxpool.json &
CUDA_VISIBLE_DEVICES=3 python ./longbench_eval.py --compress_args_path c1024_w32_k7_maxpool.json &
wait