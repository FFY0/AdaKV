from ast import arg
import sys
site_packages_path = "/home/ffy/miniconda3/envs/snapekv/lib/python3.12/site-packages/"
sys.path.insert(0,site_packages_path)
import os
import site
from datasets import load_dataset
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import time
import debugpy




import adaptive_snapkv
from adaptive_snapkv.monkeypatch.monkeypatch import  replace_mistral_fixed,replace_mistral_adaptive, replace_llama_adaptive, replace_llama_fixed

print('name_space_position:',adaptive_snapkv.__path__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("-d", '--dataset', type=str, default="THUDM/LongBench")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--skip",type=int, default=0, help="skip layer number")
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    # parser.add_argument('--adaptive', action='store_true', help="Use adaptive budgets allocation across heads")
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', 'base'], help="Ada mode, fix mode or normal")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--gqa_func',type=str, default="mean", help="gqa operation:optional max mean")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path):
    device = "cuda"
    json_data_list = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        

        if "chatglm3" in model_name_or_path:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name_or_path)
        if "chatglm3" in model_name_or_path:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[tokenizer.eos_token_id],
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name_or_path)
        json_data = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
        json_data_list.append(json_data)
        gc.collect()
        torch.cuda.empty_cache()
    with open(out_path, "w", encoding="utf-8") as f:
        for json_data in json_data_list:
            json.dump(json_data, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                            #  torch_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map="auto",
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = args.model_name_or_path
    model_name = args.model_name_or_path.split("/")[-1]
    # define your model
    max_length = args.max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        #  "multifieldqa_zh", "dureader","vcsum","lsht","passage_retrieval_zh",
        datasets = ["multi_news","qasper","narrativeqa", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum",   "trec", "triviaqa", "samsum",  \
                    "passage_count", "passage_retrieval_en",  "lcc", "repobench-p"]
        # datasets = ["qasper",]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    # NOTE: Compress config
    compress_args = {}
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress_args['floor_alpha'] = args.floor_alpha
        compress_args['gqa_support'] = args.gqa_support
        compress_args['gqa_func'] = args.gqa_func
        compress_args['normalize'] = args.normalize
        compress_args['pyram_mode']= args.pyram
        compress_args['skip'] = args.skip
        compress_args['pyram_beta'] = args.pyram_beta
        compress = True
        # if args.adaptive:
        if args.mode == "ada":
            print("Ada mode")
            replace_mistral_adaptive()
            replace_llama_adaptive()
        elif args.mode == "fix":
            print("Fix mode")
            replace_mistral_fixed()
            replace_llama_fixed()
        else:
            print("Base mode")
    else:
        print("Base mode")

    def config_compress(model, window_size=32, base_capacity=512, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, pyram_beta = 20, normalize=True, skip=0, gqa_support=False,gqa_func="mean"):
        model.model.config.window_size = window_size
        model.model.config.base_capacity = base_capacity
        model.model.config.kernel_size = kernel_size

        model.model.config.normalize = normalize
        model.model.config.pooling = pooling
        model.model.config.floor_alpha = floor_alpha

        model.model.config.pyram_mode = pyram_mode
        model.model.config.pyram_beta = pyram_beta
        model.model.config.skip = skip
        model.model.config.gqa_support = gqa_support
        model.model.config.gqa_func = gqa_func
        return model

    # NOTE: load model after replace
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # debugpy.listen(("0.0.0.0", 5678))  # 监听所有 IP 地址的 5678 端口
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client() 


    if args.compress_args_path:
        model = config_compress(model, **compress_args)
    if args.mode == "fix":
        if args.gqa_support:
            args.out_name = f"{args.out_name}_gqa_support_{args.gqa_support}_gqa_func_{args.gqa_func}"
        else:
            args.out_name = f"{args.out_name}"
    else:
        if args.gqa_support: 
            args.out_name = f"{args.out_name}_alpha_{args.floor_alpha}_gqa_support_{args.gqa_support}_gqa_func_{args.gqa_func}"
        else:
            args.out_name = f"{args.out_name}_alpha_{args.floor_alpha}"

    for dataset in datasets:
        if args.e:
            data = load_dataset(args.dataset, f"{dataset}_e", split='test', data_dir=f"{args.dataset}/data")
            if not os.path.exists(f"pred_e/{args.out_name}"):
                os.makedirs(f"pred_e/{args.out_name}")
            out_path = f"pred_e/{args.out_name}/{dataset}.jsonl"
        else:
            data = load_dataset(args.dataset, f"{dataset}", split='test', data_dir=f"{args.dataset}/data")
            if not os.path.exists(f"pred/{args.out_name}"):
                os.makedirs(f"pred/{args.out_name}")
            out_path = f"pred/{args.out_name}/{dataset}.jsonl"
            # auto skip the existing datasets   
            if os.path.exists(out_path):
                print(f"== {args.out_name} {dataset} already exists, skip this datasets")
                continue
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        # TODO: hard code single process, which use all gpus
        torch.cuda.synchronize()
        t = time.time()
        get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path)
        torch.cuda.synchronize()
        t = time.time() - t
        print(f"== {args.out_name} {dataset} Time: {t}")
