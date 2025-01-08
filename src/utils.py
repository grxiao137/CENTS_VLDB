import json
import sys
import tiktoken
import openai
import heapq
import pandas as pd
import numpy as np
from ast import literal_eval

'''
logging-related
'''
def report_warning(msg):
    print('==WARNING: ' + msg + '\n', file=sys.stderr) 


'''
tokens-related
'''
def count_tokens_w_msg(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        report_warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  
        tokens_per_name = -1 
    elif "gpt-3.5-turbo" in model:
        return count_tokens_w_msg(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return count_tokens_w_msg(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""count_tokens_w_msg() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  
    return num_tokens

def count_tokens_raw(string, model="gpt-3.5-turbo-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        report_warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(string))