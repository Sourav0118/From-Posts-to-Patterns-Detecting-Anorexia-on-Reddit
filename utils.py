import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import numpy as np
import argparse
from transformers import LongformerTokenizer, LongformerModel
from transformers import PatchTSTConfig, PatchTSTForClassification
import torch.nn as nn

def load_longformer_model(tokenizer_path, model_path):
    """Load the Longformer model and tokenizer."""
    try:
        if tokenizer_path is not None:
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", tokenizer_path)
        else:
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        if model_path is not None:
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096", model_path)
        else: 
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(Device)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Longformer model/tokenizer: {str(e)}")
        return None, None

def preprocess_text(text, tokenizer, max_len):
    """Preprocess input text for Longformer model."""
    try:
        input_ids, attn_mask = transform_single_text(text, tokenizer, max_len, max_len, 1)
        return input_ids, attn_mask
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return None, None

def get_longformer_embeddings(model, input_ids, attn_mask):
    """Generate embeddings from Longformer model."""
    model.eval()
    temp = []
    try:
        with torch.no_grad():
            for i in range(input_ids.shape[0]):
                inputs = {'input_ids': input_ids[i].unsqueeze(0).to(Device), 
                          'attention_mask': attn_mask[i].unsqueeze(0).to(Device)}
                output = model(**inputs)
                temp.append(output.last_hidden_state.squeeze().cpu())
        return torch.cat(temp, dim=0)
    except Exception as e:
        st.error(f"Error in generating Longformer embeddings: {str(e)}")
        return None

def get_time_stamps(num_time_stamps):
    """Collect and pad time series data."""
    try:
        time = []
        for _ in range(num_time_stamps):
            start = st.number_input(f"Enter start time for timestamp {_+1}", step=1)
            end = st.number_input(f"Enter end time for timestamp {_+1}", step=1)
            time.append([start, end])
        
        padded_time = np.pad(time, ((0, 2000 - len(time)), (0, 0)), 'constant')
        past_observed_mask = np.ones(2000)
        past_observed_mask[len(time):] = 0
        past_observed_mask = torch.tensor(np.stack((past_observed_mask, past_observed_mask), axis=1), dtype=torch.bool)
        return torch.tensor(padded_time, dtype=torch.float32), past_observed_mask
    except Exception as e:
        st.error(f"Error in getting time stamps: {str(e)}")
        return None, None

def load_model_weights(model, path):
    """Load pre-trained model weights."""
    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(Device)
        return model
    except Exception as e:
        st.error(f"Error loading model weights from {path}: {str(e)}")
        return None
