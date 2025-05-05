# %%
## set cache dir for hf
import os
import re
import time
import torch
import json
import math
import random
import psutil
import pickle
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
from openai import RateLimitError, APIError

from utils import *
from data.const import sotab27_ds_map as sotab_ds_map

cache_dir = ''

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# %%
if torch.cuda.is_available():
    print("GPU is available.")
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU is not available.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sotab_ds_cls = list(set(sotab_ds_map.values()))

# %% [markdown]
# ## CTA-Utils

# %%
def _clean(s: str) -> str:
    pattern = re.compile(r"[\|\\\-<>\/]")
    return pattern.sub("", s)

def refine_data(data_list):
    initial_items_count = len(data_list)
    initial_pairs_count = sum(len(item['gt']) for item in data_list)
    updated_pairs_count = 0
    ct = {}
    for data in data_list:
        original_gt_keys = list(data['gt'].keys())  
        for key in original_gt_keys:
            val = data['gt'][key].split('/')[0] if '/' in data['gt'][key] else data['gt'][key]
            if val.lower() in sotab_ds_cls:
                data['gt'][key] = val.lower()
                if data['gt'][key] in ct:
                    ct[data['gt'][key]] += 1
                else:
                    ct[data['gt'][key]] = 1
            elif val in sotab_ds_map:
                data['gt'][key] = sotab_ds_map[val]  
                if data['gt'][key] in ct:
                    ct[data['gt'][key]] += 1
                else:
                    ct[data['gt'][key]] = 1
            else:
                raise ValueError(f"Error: No mapping found for {data['gt'][key]}")

        updated_pairs_count += len(data['gt'])  
    retained_pairs_percentage = (updated_pairs_count / initial_pairs_count * 100) if initial_pairs_count else 0
    print(f"Updated {updated_pairs_count} key-value pairs ({retained_pairs_percentage:.2f}% of initial pairs).")
    print(f"Total data items processed: {initial_items_count}")
    print(ct)

def load_json_data(path):
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            tid = record.get('tid')
            tab_p = record.get('tab_p')
            c_p = record.get('c_p')
            j_p = record.get('j_p')
            gt = record.get('gt')
            results.append({
                'tid': tid,
                'tab_p': tab_p,
                'c_p': c_p,
                'j_p': j_p,
                'gt': gt
            })
    return results

def load_label(path, do_refine):
    if do_refine:
        return sotab_ds_cls
    label = set()
    cta_test_gt = pd.read_csv(path)
    for _, row in cta_test_gt.iterrows():
        label.add(row['label'])
    return list(label)

def parse_response(response, gt):
    label_batch_size = len(gt)
    if not response:
        return [''] * label_batch_size
    else:
        preds = []
        for val in list(response['response'].values()):
            if type(val) != str:
                if isinstance(val, dict):
                    while isinstance(val, dict):
                        val = list(val.values())[0]
            preds.append(val)
        if len(preds) < label_batch_size:
            preds += [''] * (label_batch_size - len(preds))
        elif len(preds) > label_batch_size:
            preds = preds[:label_batch_size]
        return preds

def parse_response_single(response, idx, sotab_ds_cls):
    k = f'Column-{idx}'
    k2 = f'column-{idx}'
    
    start = response.find('{')
    end = response.rfind('}')
    
    json_str = None
    if start != -1 and end != -1 and start < end:
        json_str = response[start:end+1]

    if json_str:
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    else:
        parsed = None

    if parsed is None:
        ## randomly choose one
        print(f'fail to parse response: {response}, returning random')
        return random.choice(sotab_ds_cls)
    else:
        k = f'Column-{idx}'
        k2 = f'column-{idx}'
        try:
            val = parsed.get(k)
            if isinstance(val, str):
                return val
    
            if val is None:
                val = parsed.get(k2)
    
            while isinstance(val, dict):
                val = list(val.values())[0]
    
            # If the value is not a string, convert it.
            if not isinstance(val, str):
                val = str(val)
        except (IndexError, KeyError, TypeError):
            print(f'fail to parse response: {response}, returning random')
            val = random.choice(sotab_ds_cls)
    
    return val

def find_in_labels(response, sotab_ds_cls):
    for cls in sotab_ds_cls:
        if cls in response:
            return cls
    return None

def parse_response_single_find(response, idx, sotab_ds_cls):
    k = f'Column-{idx}'
    k2 = f'column-{idx}'
    
    start = response.find('{')
    end = response.rfind('}')
    
    json_str = None
    if start != -1 and end != -1 and start < end:
        json_str = response[start:end+1]

    if json_str:
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    else:
        parsed = None

    if parsed is None:
        cls = find_in_labels(response, sotab_ds_cls)
        if not cls:
            ## randomly choose one
            print(f'fail to parse response: {response}, returning random')
            return random.choice(sotab_ds_cls)
        else:
            return cls
    else:
        k = f'Column-{idx}'
        k2 = f'column-{idx}'
        try:
            val = parsed.get(k)
            if isinstance(val, str):
                return val
    
            if val is None:
                val = parsed.get(k2)
    
            while isinstance(val, dict):
                val = list(val.values())[0]
            if not isinstance(val, str):
                val = str(val)
        except (IndexError, KeyError, TypeError):
            cls = find_in_labels(response, sotab_ds_cls)
            if not cls:
                ## randomly choose one
                print(f'fail to parse response: {response}, returning random')
                return random.choice(sotab_ds_cls)
            else:
                return cls
    
    return val

# %% [markdown]
# ## Dataloading & processing

# %%
## Load Data
######################################################################
b2000_data_path = ''
b4000_data_path = ''
b6000_data_path = ''
b8000_data_path = ''

# %%
b2000_out_dir = ''
b4000_out_dir = ''
b6000_out_dir = ''
b8000_out_dir = ''

# %%
## TODO
datadir = b2000_data_path
ourdir = b2000_out_dir
budget = 2000
max_new_tokens = 4096

# %%
data = load_json_data(datadir)

# %%
## Load Label
label_path = ''
label_space = load_label(label_path, True)
label_space
######################################################################

# %%
refine_data(data)

# %% [markdown]
# ## Deepseek-R1-Distill-Llama-8B

# %%
## deep-seek-r1 init
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %% [markdown]
# ## Column Type Annotation


# %%
ntables = len(data)
shard_size = 500
shards = math.ceil(ntables / shard_size)

# %%
processed_id = set()
g_ct = {}
_temp_t = []
_max_t = 0
_max_t_id = ''
_ttt=0

# %%
def gen_response(model, tokenizer, prompt, max_new_tokens, parse_thinking=True, verbose=False):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    if verbose:
        print(f'Generated_text (with reasoning): {generated_text}')
        
    if parse_thinking:
        marker = '</think>'
        marker_index = generated_text.find(marker)
        generated_text = generated_text[marker_index + len(marker):]
        if verbose:
            print(f'Generated_text (w/o reasoning): {generated_text}')
    return generated_text

# %%
for i in range(shards):
    data_shard = data[i*shard_size: (i+1)*shard_size]
    p = f'deepseek-r1-CENTS-llama-8B-{ntables}-budget({budget})-shard{i}-SOTAB.json'
    outpath = os.path.join(ourdir, p)
    gt_shard, pred_shard, time_shard  = {}, {}, {}
    for j in range(len(data_shard)):
        tid = data_shard[j]['id']
        if tid in processed_id:
            raise ValueError(f'Redundant tid: {tid} detected in shards-{i} and {j}th item')
        processed_id.add(tid)
        tab_p, c_p, j_p, gt = data_shard[j]['tab_p'], data_shard[j]['c_p'],data_shard[j]['j_p'], data_shard[j]['gt']
        
        st = time.time()
        pred = []
        assert len(c_p) == len(j_p), "len of cp must == jp"
        history = [tab_p]
        for i in range (len(j_p)):
            new_msg = [c_p[i], j_p[i]]
            this_msg = history + new_msg
            this_msg_str = '\n'.join(this_msg)
            response = gen_response(model, tokenizer, this_msg_str, max_new_tokens, parse_thinking=True, verbose=False)
            this_pred = parse_response_single_find(response, i, sotab_ds_cls)
            pred.append(this_pred)

        et = time.time()
        gt_shard[tid] = gt
        pred_shard[tid] = pred
        time_shard[tid] = et - st

    print(f'saving to {outpath}')
    with open(outpath, 'w') as f:
        res = {'gt': gt_shard, 'pred': pred_shard, 'time': time_shard}
        json.dump(res, f)


