import re
import warnings
import os
import json
import time
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
from data.const import sotab41_ds_map as sotab_ds_map
from src.cdr.message import Message, MessageRole
sotab_ds_cls = list(set(sotab_ds_map.values()))
from src.connector import gpt_connector as Connector
from src.serializer import dfser as Serializer

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
            data.append(t)
    return data

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

def parse_response_annotate_full(response, gt, q_col_idx):
    label_batch_size = len(gt)
    if not response:
        return [''] * label_batch_size
    else:
        preds = []
        for idx in q_col_idx:
            try:
                val = list(response['response'].values())[idx]
                while isinstance(val, dict):
                    val = list(val.values())[0]
                if not isinstance(val, str):
                    val = str(val)
            except (IndexError, KeyError, TypeError) as e:
                val = ''

            preds.append(val)
        return preds

def parse_response_annotate_partial(response, gt, q_col_idx):
    label_batch_size = len(gt)
    if not response:
        return [''] * label_batch_size
    else:
        preds = []
        for i in range(len(q_col_idx)):
            k = f'Column-{i}'
            k2 = f'column-{i}'
            try:
                val = response['response'].get(k)
                if val is None:
                    val = response['response'].get(k2)

                while isinstance(val, dict):
                    val = list(val.values())[0]

                if not isinstance(val, str):
                    val = str(val)
            except (IndexError, KeyError, TypeError) as e:
                val = ''
            preds.append(val)

        if len(preds) < label_batch_size:
            preds += [''] * (label_batch_size - len(preds))
        elif len(preds) > label_batch_size:
            preds = preds[:label_batch_size]

        return preds

def parse_response_single(response, gt, idx):
    gold = False
    ## no response received here, random gen
    if not response or not response['response']:
        return random.choice(sotab_ds_cls), gold
        # return ''
    else:
        val = ''
        if f"column-{idx+1}" in str(response['response'].keys()).lower() and \
                "column-0" in str(response['response'].keys()).lower():
            try:
                val = list(response['response'].values())[0]
                if isinstance(val, str):
                    gold = True
                    return val, gold

                while isinstance(val, dict):
                    val = list(val.values())[0]

                if not isinstance(val, str):
                    val = str(val)
            except (IndexError, KeyError, TypeError) as e:
                val = ''

        if val == '':
            try:
                val = list(response['response'].values())[0]
                while isinstance(val, dict):
                    val = list(val.values())[0]
                while isinstance(val, list):
                    val = val[0]
                if not isinstance(val, str):
                    val = str(val)
            except (IndexError, KeyError, TypeError) as e:
                val = ''

        return val, gold

def re_message(df, qcol_ind, label_space, ser):
    p = f'''You are an expert in the field of column type annotation and understanding tabular data. Your task is to find semantic relationships of pairs of columns of a table.'''
    sys_p = [Message(MessageRole.SYS, p)]
    qcol_names = [df.columns[i] for i in qcol_ind]
    new_order = qcol_names + [col for col in df.columns if col not in qcol_names]
    df_reordered = df[new_order]
    df_reordered.columns = range(len(df_reordered.columns))
    df_reordered.reset_index(drop=True, inplace=True)
    ser_context = ser.serialize(df_reordered, si = 0)
    user_p = []
    up = f'''
            Given the following table: \n
            {ser_context} \n

            Your instructions are: 1. Look at the input given to you in a dataframe format and make a table out of it. 
            2. Look at the cell values in detail and understand them. 
            '''
    user_p.append(Message(MessageRole.USER, up))
    for i, _ in enumerate(qcol_ind):
        subj_column = df_reordered.iloc[:,0].astype(str)
        subj_column_list = subj_column.tolist()
        this_label_space = label_space
        ser_labels = ', '.join(this_label_space)
        p = f'''
        3. For Column-{i+1} in the table, determine a SINGLE semantic relationship with Column-0. You should be confident if you use a class from [{ser_labels}], otherwise, create a new class based on your understanding that best fits the relationship between Column-{i+1} and Column-0, and this new class should be specific.
        4. Return a JSON format answer following the below instruction.
        '''
        user_p.append(Message(MessageRole.USER, p))
    json_p = []
    for i in range(len(qcol_ind)):
        json_obj = {'Reason': "YOUR REASON", f'Column-{i}': "YOUR ANSWER"}
        p = '''Following is the JSON object you need to fill with your answers with. Return a complete and parsable JSON object.\n'''
        p += f'''{json.dumps(json_obj, indent=4)}'''
        json_p.append(Message(MessageRole.USER, p))
    return sys_p, user_p, json_p


def run(args):
    budget = args.budget

    # load data
    if not os.path.exists(args.data) or str(budget) not in args.data:
        raise FileNotFoundError(f"Data file {args.data} not found or budget mismatch.")
    else:
        data = load_json_data(args.data)

    refine_data(data)
    
    if not os.path.exists(args.label):
        raise FileNotFoundError(f"Label file {args.label} not found.")
    else:
        label_space = load_label(args.label, args.use_sub_label)

    if not args.ntables or args.ntables > len(data):
        ntables = len(data)
    else:
        ntables = args.ntables
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    conn = Connector(model_name=args.model, model_api='')
    serializer = Serializer()

    print(f'===' * 50)
    print(f'Starting RE')
    print(f'Arguments: {args}')
    print(f'===' * 50)

    shard_size = args.shard_size

    shards = math.ceil(ntables / shard_size)

    processed_id = set()

    _temp_t = []

    for i in range(shards):
        data_shard = data[i*shard_size: (i+1)*shard_size]
        p = f'{args.model}-budget{budget}-shard{i}-RE.json'
        outpath = os.path.join(args.outdir, p)

        gt_shard, pred_shard, time_shard  = {}, {}, {}
        for j in range(len(data_shard)):
            tid = data_shard[j]['id']
            if tid in processed_id:
                raise ValueError(f'Redundant tid: {tid} detected in shards-{i} and {j}th item')
            processed_id.add(tid)

            df, _tok, gt = data_shard[j]['table'], data_shard[j]['tokens'], data_shard[j]['gt']
            q_col_idx, q_col_gt = [int(gi) for gi in list(gt.keys())], list(gt.values())
            q_col_idx, q_col_gt = map(list, zip(*sorted(zip(q_col_idx, q_col_gt))))
            _temp_t.append(_tok)

            ## TODO: Sort col idx
            df = df.sort_index(axis=1)

            if args.verbose:
                if True:
                    print(df.head(10))
                    print(q_col_idx)
                    print(q_col_gt)
                    for idx, gt in zip(q_col_idx, q_col_gt):
                        print('*' * 20)
                        this_column = df.iloc[:,idx].astype(str)
                        print(this_column)
                        print(gt)

            if len(df.columns) == 0:
                print('no df')
                gt_shard[tid] = q_col_gt
                pred_shard[tid] = [random.choice(sotab_ds_cls) for i in range(len(q_col_gt))]
                time_shard[tid] = 0
                continue

            s_msgs, p_msgs, json_msgs = re_message(df, q_col_idx, label_space, serializer)

            pred = []
            history = s_msgs + [p_msgs[0]]
            st = time.time()
            for i in range (len(json_msgs)):
                new_msg = [p_msgs[i+1], json_msgs[i]]
                this_msg = history + new_msg
                response = conn.submit(msgs=this_msg, count_tokens=False, verbose=args.verbose, retry=3)
                this_pred, _ = parse_response_single(response, q_col_gt[i], i)
                pred.append(this_pred)
                if this_pred:
                    p = f'''{{"Column-{i+1}'s semantic relationship with Column-{0}": "{this_pred}"}}'''
                    history = history + [json_msgs[i]] + [Message(MessageRole.ASSISTANT, p)]

            et = time.time()
            for qcidx in q_col_idx:
                unique_values = df.iloc[:, qcidx].apply(str).unique()
                if ('None' in unique_values or 'nan' in unique_values) and len(unique_values) <= 1:
                    pred[q_col_idx.index(qcidx)] = random.choice(sotab_ds_cls)

            gt_shard[tid] = q_col_gt
            pred_shard[tid] = pred
            time_shard[tid] = et - st
        
            if (j+1) % 250 == 0 and j != 0:
                print(f'Processed {j} tables')

        # save
        with open(outpath, 'w') as f:
            res = {'gt': gt_shard, 'pred': pred_shard, 'time': time_shard, 'args': [(key, value) for key, value in vars(args).items()]}
            json.dump(res, f)
        print('*'*100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RE task of L(L)M (for) T(able Understanding)')
    # core
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--label', type=str, help='Path to the label file')
    parser.add_argument('--outdir', type=str, help='Dir to save the output')
    parser.add_argument('--ntables', type=int, help='Number of tables to inference')
    args = parser.parse_args()
    run(args)