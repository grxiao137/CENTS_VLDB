import pickle
import sys
import os
import re
import json
import pandas as pd
import numpy as np
import tiktoken
import fasttext
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
from src.cdr import OMHyb, OMCS
from src.serializer import dfser as DFSerializer

def count_tokens(text):
    return len(tokenizer.encode(text))

def clean_text(cell):
    if isinstance(cell, str):
        cleaned_text = re.sub(' +', ' ', str(cell)).strip() 
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cleaned_text, tokens_count
    elif isinstance(cell, list):
        cleaned_list = []
        tokens_count = 0
        for item in cell:
            cleaned_item, item_tokens = clean_text(item)
            cleaned_list.append(cleaned_item)
            tokens_count += item_tokens
        return cleaned_list, tokens_count
    elif isinstance(cell, dict):
        cleaned_dict = {}
        tokens_count = 0
        for key, value in cell.items():
            cleaned_value, value_tokens = clean_text(value)
            cleaned_dict[key] = cleaned_value
            tokens_count += value_tokens
        return cleaned_dict, tokens_count
    elif isinstance(cell, (int, float)):
        cleaned_text = str(cell)
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cell, tokens_count 
    else:
        cleaned_text = re.sub(' +', ' ', str(cell)).strip()
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cleaned_text, tokens_count


cta_test_gt = pd.read_csv('data/SOTAB/raw/CTA/CTA_Test/CTA_test_gt.csv')
gt = {'train':{}, 'val':{}, 'test':{}}
test = {}
for index, row in cta_test_gt.iterrows():
    if row['table_name'] not in gt['test']:
        gt['test'][row['table_name']] = {}
    gt['test'][row['table_name']][row['column_index']] = row['label']

def read_table(fp):
    if fp.endswith('.json.gz'):
        t = pd.read_json(fp, compression='gzip', lines=True)
        cleaned_data = t.applymap(clean_text)
        cleaned_text_df = cleaned_data.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
        token_count_df = cleaned_data.applymap(lambda x: x[1] if isinstance(x, tuple) else x)
        total_tokens = token_count_df.sum().sum()
        
        return cleaned_text_df, total_tokens
    return None, None

test = {}
tokens = {}

for file in os.listdir('data/SOTAB/raw/CTA/CTA_Test/Test/'):
    df, ct = read_table(f'data/SOTAB/raw/CTA/CTA_Test/Test/{file}')
    if len(df) < _min:
        _min = len(df)
    if len(df) > _max:
        _max = len(df)
    if df is not None and file in gt['test']:
        test[file] = {'table': df, 'tokens': ct, 'gt': gt['test'][file]}
        tokens[file] = ct
label_set = list(cta_test_gt['label'].unique())

print(f'len(test): {len(test)}, len(label_set): {len(label_set)}')

def sample_numerical_and_datetime_columns(df):
    nnd_data = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        clean_series = df[col].dropna()  
        if len(clean_series) == 0:
            nnd_data[col] = []
        else:
            num_samples = min(3, len(clean_series))  
            selected_cells = clean_series.sample(num_samples).tolist()  
            nnd_data[col] = selected_cells
        df = df.drop(col, axis=1)  
    for col in df.select_dtypes(include=['datetime']).columns:
        orig_series = df[col]
        num_samples = min(3, len(orig_series))
        selected_cells = clean_series.tolist()[:num_samples]  
        nnd_data[col] = selected_cells
        df = df.drop(col, axis=1)  
    return df, nnd_data


def restore_numerical_columns(df, numerical_data, original_columns):
    if not numerical_data:
        return df
    for col, cells in numerical_data.items():
        df[col] = pd.Series(cells)
    return df[original_columns]

## EE-Budget Allocation
def calculate_empirical_entropy(column):
    valid_cells = column.dropna().astype(str)
    valid_cells = [cell for cell in valid_cells if cell.lower() not in ["nan", "none"]]
    unique_toks = {}
    for cell in valid_cells:
        cell_tokens = tokenizer.encode(cell)
        for ct in cell_tokens:
            unique_toks[ct] = unique_toks.get(ct, 0) + 1
    total_tokens = sum(unique_toks.values())
    if total_tokens == 0:
        return 0 
    probabilities = [count / total_tokens for count in unique_toks.values()]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_column_budgets(df, total_budget):
    entropies = {}
    _f = False
    all_e = []
    for col in df.columns:
        entropy = calculate_empirical_entropy(df[col])
        if entropy == 0:
            _f = True
        else:
            all_e.append(entropy)
        entropies[col] = entropy

    if not _f:
        total_entropy = sum(entropies.values())
        c2b = {col: total_budget * (entropy / total_entropy) for col, entropy in entropies.items()}
    else:
        total_entropy = sum(entropies.values())
        c2b = {col: total_budget * (entropy / total_entropy) if entropy!= 0 else total_budget * (min(all_e) / total_entropy) for col, entropy in entropies.items()}

    return c2b


ser = DFSerializer()
budget = [2000, 4000, 6000, 8000]
results_dir = ''
results_paths = [results_dir + '{}.data'.format(b) for b in budget]

ft_model_path = 'data/models/cc.en.100.bin'
scorer = OMHyb(tokenizer=tokenizer, model_path=ft_model_path)
solvers = [OMCS()]
num_ct = 0

for solver in solvers:
    for b in budget:
        max_token, max_time = 0, 0
        sampled_data = []
        sampled_table, total_time_score, total_time_sel = 0, 0, 0
        total_score = 0
        print(f'Working on budget: {b} solver {solver}')
        for table_name, data in list(test.items()):
            df, tokens, gt = data['table'], data['tokens'], data['gt']
            captions = None if 'caption' not in data else data['caption']
            original_columns = df.columns.tolist()  
            if tokens > b:
                numerical_data = None
                df, numerical_data = sample_numerical_and_datetime_columns(df)
                num_ct += len(numerical_data.keys())
                tnd = count_tokens(str(numerical_data))
                c2b = calculate_column_budgets(df, b-tnd)
                c2c, c2s, c2w = scorer.gen_score(df, c2b, gt)
                sampled_table += 1
                c2sel, this_score, time_taken = solver.solve(c2c, c2w, c2s, c2b, b, verbose=False)
                max_length = max([len(values) for values in c2sel.values()], default=0)
                reconstructed_df = pd.DataFrame({col: c2sel.get(col, []) + [None] * (max_length - len(c2sel.get(col, []))) for col in original_columns})
                reconstructed_df = restore_numerical_columns(reconstructed_df, numerical_data, original_columns)
                assert reconstructed_df.columns.tolist() == original_columns, "Column order mismatch"
                df = reconstructed_df.to_json(orient='records')
            else:
                df = df.to_json(orient='records')
            item = {'id': table_name, 'table': df, 'tokens': str(tokens), 'gt': gt, 'time_taken': str(time_taken)}
            if captions:
                item['caption'] = captions
            sampled_data.append(item)

        with open(results_paths[budget.index(b)], 'w') as f:
            for item in sampled_data:
                f.write(json.dumps(item) + '\n')

