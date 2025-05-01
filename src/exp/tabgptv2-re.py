# %%
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

from utils import *
from data.const import sotab41_ds_map as sotab_ds_map
from src.cdr.message import Message, MessageRole
sotab_ds_cls = list(set(sotab_ds_map.values()))
from src.connector import gpt_connector as Connector
from src.serializer import dfser as Serializer
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

# %% [markdown]
# ## RE-Utils

# %%
def _clean(s: str) -> str:
    pattern = re.compile(r"[\|\\\-<>\/]")
    return pattern.sub("", s)


def load_label(path, do_refine):
    if do_refine:
        return sotab_ds_cls
    label = set()
    cta_test_gt = pd.read_csv(path)
    for _, row in cta_test_gt.iterrows():
        label.add(row['label'])
    return list(label)

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


def find_in_labels(response, sotab_ds_cls):
    for cls in sotab_ds_cls:
        if cls in response:
            return cls
    return None

def extract_first_json(response: str):
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(response):
        if ch in ('{'):
            try:
                # try decoding from this index
                obj, end = decoder.raw_decode(response, idx)
                return obj
            except json.JSONDecodeError:
                # not valid JSON at this brace—keep looking
                continue
    return None


def parse_response_single_find(response, idx, sotab_ds_cls):
    parsed = extract_first_json(response)
    if parsed is None:
        cls = find_in_labels(response, sotab_ds_cls)
        if not cls:
            ## randomly choose one
            print(f'fail to parse response: {response}, returning random')
            return random.choice(sotab_ds_cls)
        else:
            return cls
    else:
        # print(f'parsed.values: {parsed.values()}')
        try:
            val = list(parsed.values())[0]
            # If the value is not a string, convert it.
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



# %%
## TODO
datadir = None
ourdir = None 
budget = 2000
max_new_tokens = 4096

# %%
data = load_json_data(datadir)

# %%
## Load Label
label_path = None
label_space = load_label(label_path, True)
label_space


# %% [markdown]
# ## Main

# %%
## tab-gpt
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "tablegpt/TableGPT2-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,)

# %%
refine_data(data)
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
def gen_response(model, tokenizer, prompt, max_new_tokens, verbose=False):
   
    messages = [
    {"role": "system", "content": "You are a helpful assistant that is good at Relation Extraction for tables. You can identify accurately the semantic relationship between columns."},
    {"role": "user", "content": prompt},]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if verbose:
         print(f'Prompt: {text}')

    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # inputs = tokenizer(prompt, return_tensors="pt")
    # inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    # outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if verbose:
        print(f'Generated_text: {response}')
        
    return response

# %%
for i in range(shards):
    data_shard = data[i*shard_size: (i+1)*shard_size]
    p = f'tabgptv2-RE-{ntables}-budget({budget})-shard{i}-SOTAB.json'
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
            response = gen_response(model, tokenizer, this_msg_str, max_new_tokens, verbose=False)
            this_pred = parse_response_single_find(response, i, sotab_ds_cls)
            pred.append(this_pred)
                
        et = time.time()
        gt_shard[tid] = gt
        pred_shard[tid] = pred
        time_shard[tid] = et - st

    with open(outpath, 'w') as f:
        res = {'gt': gt_shard, 'pred': pred_shard, 'time': time_shard}
        json.dump(res, f)





