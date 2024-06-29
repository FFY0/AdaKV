
conda activate snapekv
#os.environ['project_dir'] = 'D:\Code_repository\LLM\AdaptiveSnape'
#os.environ['datasets_dir'] = 'D:\Code_repository\ShareFiles\Datasets'
#os.environ['models_dir'] = 'D:\Code_repository\ShareFiles\Models'
export project_dir=/tmp/pycharm_project_148
export datasets_dir=/raid/fengyuan/datasets
export models_dir=/raid/share_files/models

CUDA_VISIBLE_DEVICES=2 python ./longbench_eval.py --adaptive --floor 0.5  --normalize --compress_args_path c1024_w32_k7_maxpool.json &
CUDA_VISIBLE_DEVICES=3 python ./longbench_eval.py --adaptive --floor 0.5  --compress_args_path c1024_w32_k7_maxpool.json &
wait
