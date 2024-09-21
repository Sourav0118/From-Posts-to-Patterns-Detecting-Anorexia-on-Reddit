from torch import Tensor
import torch
from transformers import PreTrainedTokenizerBase
import re
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerModel
from transformers import AutoTokenizer
import random
from datetime import datetime

def clean_text(text):
    # Remove URLs
    text = re.sub(R"https?://\S+|www\.\S+", "", text)
    # Remove brackets (both round and square brackets)
    text = re.sub(r"[\(\)\[\]\{\}]", "", text)
    # Remove punctuation marks and non-word characters
    text = re.sub(r"[^\w\s]", "", text)
    return text


def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            filename, target = line.strip().split()
            data.append((filename, int(target)))
    return data


def split_overlapping(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int):
    input_ids = tensor["input_ids"].squeeze(0)
    attn_mask = tensor["attention_mask"].squeeze(0)
    result_input_id = [input_ids[i: i+chunk_size] for i in range(0, len(input_ids), stride)]
    result_attn_mask = [attn_mask[i: i+chunk_size] for i in range(0, len(attn_mask), stride)]
    if len(result_input_id) > 1:
        result_input_id = [x for x in result_input_id if len(x) >= minimal_chunk_length]
        result_attn_mask = [x for x in result_attn_mask if len(x) >= minimal_chunk_length]
    return result_input_id, result_attn_mask


def add_special_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]):
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])


def add_padding(input_id_chunks: list[Tensor], mask_chunks: list[Tensor], chunk_size: int):
    for i in range(len(input_id_chunks)):
        pad_len = chunk_size + 2 - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0]*pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0]*pad_len)])


def stack_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]):
    input_ids = torch.stack(input_id_chunks)
    attn_mask = torch.stack(mask_chunks)
    return input_ids.long(), attn_mask.int()


def transform_single_text(text: str, tokenizer: PreTrainedTokenizerBase,
                          chunk_size: int, stride: int, minimal_chunk_length: int):
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
    input_id_chunks, mask_chunks = split_overlapping(tokens, chunk_size, stride, minimal_chunk_length)
    add_special_tokens(input_id_chunks, mask_chunks)
    add_padding(input_id_chunks, mask_chunks, chunk_size)
    input_ids, attention_mask = stack_tokens(input_id_chunks, mask_chunks)
    return input_ids, attention_mask


def augment_dataframe(df, num_augmentations=1):
    augmented_data = []
    for index, row in df.iterrows():
        if row['Anorexia'] == 1:
            augmented_texts = augment_sentence(clean_text((row['text']).lower()), num_augmentations)
            for text in augmented_texts:
                augmented_data.append({'subject_ID': row['subject_ID'], 
                                       'text': text, 
                                       'Anorexia': row['Anorexia']})
    
    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    # Append augmented data to the original DataFrame
    new_df = pd.concat([df, augmented_df], ignore_index=True)
    
    return new_df


def clean_text(text):
    # Remove URLs
    text = re.sub(R"https?://\S+|www\.\S+", "", text)
    # Remove brackets (both round and square brackets)
    text = re.sub(r"[\(\)\[\]\{\}]", "", text)
    # Remove punctuation marks and non-word characters
    text = re.sub(r"[^\w\s]", "", text)
    return text

def preprocess_text(text, tokenizer, max_len):
    """Preprocess input text for Longformer model."""
    try:
        input_ids, attn_mask = transform_single_text(text, tokenizer, max_len, max_len, 1)
        return input_ids, attn_mask
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return None, None
