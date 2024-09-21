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
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if tokenizer_path != None:
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", tokenizer_path)
        else:
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        if model_path != None:
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096", model_path)
        else: 
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(Device)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading Longformer model/tokenizer: {str(e)}")
        return None, None

def load_model_weights(model, path):
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    """Load pre-trained model weights."""
    try:
        model.load_state_dict(torch.load(path, map_location = torch.device(Device)))
        model.to(Device)
        return model
    except Exception as e:
        print(f"Error loading model weights from {path}: {str(e)}")
        return None
