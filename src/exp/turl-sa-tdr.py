import re
import os
import json
import time
import tiktoken
import math
import torch
import random
import psutil
import pickle
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter

from src.connector import gpt_connector as Connector
from src.serializer import dfser as Serializer
from src.cdr.message import Message, MessageRole

def gen_message(df, label_space, ser):
    # sys_p
    ser_labels = ', '.join(label_space)
    p1 = ''
    p1 += f'''You are an expert in analyzing tabular data and recommending the most relevant headers (attributes) based on the table's content and existing headers. Your deep understanding of data structures and attribute inference is crucial for this task.''' 
    p1 += f'''Given the following list of potential new headers: {ser_labels}, your task is to rank these headers from most to least relevant, considering the existing headers, cell contents, and structure of the table.'''
    p1 += f'''Please reply with only the top 10 ranked headers but nothing else (e.g., do not provide explanations). Make sure all given headers exist in your response in ranked order.'''
    p1 += f'''Please provide your answer in JSON format, listing the headers in order of relevance.'''

    sys_p = [Message(MessageRole.SYS, p1)]

    ser_context = ser.serialize(df, item_sep='sa')
    p = f'''
        Given the following table in dataframe format: \n
        {ser_context} \n
        
        Your task is to:
        1. Analyze the provided table, focusing on the captions, the existing column headers, and content of the cell values.
        2. Carefully examine the table to understand the context.
        3. From the given set of possible column headers, rank all the headers in order of their relevance for augmenting the table schema and content.

        Please provide your answer in a ranked list, based on the relevance of each header to the existing data.
    '''
    user_p = [Message(MessageRole.USER, p)]
    # json_p
    json_p = []
    p = {}
    p[f'Headers_from_most_relevant_to_least_relevant'] = 'YOUR ANSWER HERE'
    json_p.append(Message(MessageRole.USER, json.dumps(p, indent=4)))
    return sys_p, user_p, json_p

def create_dataframe(tableData, processed_tableHeaders):
    rows = []
    for row in tableData:
        row_data = [cell['text'] if cell['text'] else None for cell in row]
        rows.append(row_data)
    
    dataframe = pd.DataFrame(rows, columns=processed_tableHeaders)
    
    return dataframe

def load_map(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    original_phrase, representative_phrase = parts
                    mapping[original_phrase] = representative_phrase
                else:
                    print(f"Warning: Line format incorrect: '{line.strip()}'")
    return mapping

def load_json_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            t = json.loads(line)
            if isinstance(t['table'], str):
                # Wrap the JSON string in a StringIO object
                table_io = StringIO(t['table'])
                t['table'] = pd.read_json(table_io)
            elif isinstance(t['table'], dict):
                t['table'] = pd.DataFrame(t['table'])
            else:
                raise ValueError("Unexpected data type for 'table'. Expected JSON string or dictionary.")
            t['tokens'] = int(t['tokens'])
            t['caption'] = t["caption"] if "caption" in t else None
            data.append(t)
    return data

def load_label(path):
    with open(path) as f:
        task_data = f.read().splitlines() 
    return task_data

def parse_response_rank(response, gt, topklabels):
    if not response:
        return [''] * gt
    try:
        preds = list(response['response'].values())[0]
        if not isinstance(preds, list):
            print(f'''preds not list {preds}''')
            return topklabels 
        return preds
    except:
        return topklabels 

def run(args):
    if args.path_reduc_labels:
        assert args.topk in [10, 15, 20, 30, 50, 100, 200, 1000], "Top-k value for reducer is not supported."
        reduced_labels = pickle.load(open(args.path_reduc_labels, 'rb'))

    data = load_json_data(args.data)
    if args.use_sub_label:
        down_sample_map = load_map(args.use_sub_label)

    if not os.path.exists(args.label):
        raise FileNotFoundError(f"Label file {args.label} not found.")
    else:
        label_space = load_label(args.label)

    # check number of tables
    if not args.ntables or args.ntables > len(data):
        ntables = len(data)
    else:
        ntables = args.ntables

    # check output path
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    

    random.shuffle(data)
    conn = Connector(model_name=args.model, model_api='')
    serializer = Serializer()

    print(f'===' * 50)
    print(f'Starting sa task of L(L)M (for) T(able Understanding)')
    print(f'Arguments: {args}')
    print(f'===' * 50)

    shard_size = args.shard_size

    shards = math.ceil(ntables / shard_size)

    processed_id = set()

    _temp_t = []
    _orig_label_space = label_space
    for i in range(shards):
        data_shard = data[i*shard_size: (i+1)*shard_size]
        p = f'{args.model}-budget{args.budget}-shard{i}-sa.json'
        outpath = os.path.join(args.outdir, p)

        gt_shard, pred_shard, time_shard  = {}, {}, {}
        for j in range(len(data_shard)):
            tid = data_shard[j]['id']
            if tid in processed_id:
                raise ValueError(f'Redundant tid: {tid} detected in shards-{i} and {j}th item')
            processed_id.add(tid)

            df, _tok, gt = data_shard[j]['table'], data_shard[j]['tokens'], data_shard[j]['gt']
            caption = None
            _temp_t.append(_tok)

            df = df.sort_index(axis=1)
            label_space = reduced_labels.get(args.topk, {}).get(tid, None)
            if not label_space: 
                continue
            label_space = label_space[0][0]
            gt = reduced_labels.get(args.topk, {}).get(tid, None)[0][1]
            for i, orig_label in enumerate(label_space):
                # TODO: temp fix
                if orig_label in down_sample_map:
                    label_space[i] = down_sample_map[orig_label]
                elif orig_label + ' ' + orig_label in down_sample_map:
                    label_space[i] = down_sample_map[orig_label + ' ' + orig_label]
                else:
                    print('no match, fall defaul')
                    label_space[i] = orig_label 
            for i, orig_label in enumerate(gt):
                gt[i] = down_sample_map[orig_label]

            for c in gt:
                if c in df.columns:
                    df.drop(c, axis=1, inplace=True)
            if len(df.columns) == 0 or len(gt) == 0:
                print("df has no columns left or no gt")
                continue

            old_label_space = label_space
            label_space = []
            for c in old_label_space:
                if c in df.columns:
                    continue
                else:
                    label_space.append(c)

            s_msgs, p_msgs, json_msgs = gen_message(df, label_space, serializer)

            pred = []
            st = time.time()
            this_msg = s_msgs + p_msgs + json_msgs
            response = conn.submit(msgs=this_msg, count_tokens=False, verbose=args.verbose, retry=3)
            pred = parse_response_rank(response, gt, label_space)

            et = time.time()

            gt_shard[tid] = gt
            pred_shard[tid] = pred
            time_shard[tid] = et - st

        if args.verbose:
            print('gt:')
            print(gt_shard[tid])
            print('pred:')
            print(pred_shard[tid])
    
        if (j+1) % 500 == 0 and j != 0:
            print(f'Processed {j} tables')

        # save
        with open(outpath, 'w') as f:
            res = {'gt': gt_shard, 'pred': pred_shard, 'time': time_shard, 'args': [(key, value) for key, value in vars(args).items()]}
            json.dump(res, f)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SA task of L(L)M (for) T(able Understanding)')
    # core
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--label', type=str, help='Path to the label file')
    parser.add_argument('--outdir', type=str, help='Dir to save the output')
    parser.add_argument('--ntables', type=int, help='Number of tables to inference')
    parser.add_argument('--topk', type=int, help='top-k for task reduc', default=10)
    parser.add_argument('--path_reduc_labels', type=str, required=False, help='Path to the reduced label file, per col, per table')
    parser.add_argument('--use_sub_label', type=str, required=False, help='Path to the downsample map')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output', default=False)


    args = parser.parse_args()
    run(args)