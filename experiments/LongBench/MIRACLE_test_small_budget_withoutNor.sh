
conda activate snapekv
#os.environ['project_dir'] = 'D:\Code_repository\LLM\AdaptiveSnape'
#os.environ['datasets_dir'] = 'D:\Code_repository\ShareFiles\Datasets'
#os.environ['models_dir'] = 'D:\Code_repository\ShareFiles\Models'
export project_dir=/xikexie/AdaptiveSnape
export datasets_dir=/xikexie/66ring/datasets
export models_dir=/xikexie/66ring/models

CUDA_VISIBLE_DEVICES=0 python ./longbench_eval.py --adaptive --floor 0.5  --compress_args_path c256_w32_k7_maxpool.json &
CUDA_VISIBLE_DEVICES=1 python ./longbench_eval.py --adaptive --floor 0.5  --compress_args_path c512_w32_k7_maxpool.json &
wait
