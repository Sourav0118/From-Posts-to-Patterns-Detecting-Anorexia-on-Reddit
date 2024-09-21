import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import os
import torch
import numpy as np
import streamlit as st
from transformers import LongformerTokenizer, LongformerModel
from transformers import PatchTSTConfig, PatchTSTForClassification
import torch.nn as nn
from preprocessing import *
from model import *

# Initialize necessary components
sigmoid = nn.Sigmoid()
Device = "cuda" if torch.cuda.is_available() else "cpu"

config = PatchTSTConfig(
    num_input_channels=2,
    num_targets=1,
    context_length=2000,
    patch_length=12,
    stride=12,
    use_cls_token=True,
    head_dropout=0.5,
)

def load_longformer_model(tokenizer_path, model_path):
    """Load the Longformer model and tokenizer."""
    try:
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=tokenizer_path)
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096", cache_dir=model_path)
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
    time = []
    for i in range(num_time_stamps):
        start = st.number_input(f"Enter Hr for timestamp {i+1}", step=1)
        end = st.number_input(f"Enter Min for timestamp {i+1}", step=1)
        time.append([start, end])
    
    padded_time = np.pad(time, ((0, 2000 - len(time)), (0, 0)), 'constant')
    past_observed_mask = np.ones(2000)
    past_observed_mask[len(time):] = 0
    past_observed_mask = torch.tensor(np.stack((past_observed_mask, past_observed_mask), axis=1), dtype=torch.bool)
    return torch.tensor(padded_time, dtype=torch.float32).unsqueeze(0), past_observed_mask.unsqueeze(0)

def load_model_weights(model, path):
    """Load pre-trained model weights."""
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device(Device)))
        model.to(Device)
        return model
    except Exception as e:
        st.error(f"Error loading model weights from {path}: {str(e)}")
        return None

# Streamlit app starts here
st.title("Anorexia Prediction")

# Input parameters
input_text = st.text_area("Enter the input text for the model")
num_time_stamps = st.number_input("Enter the number of time stamps", min_value=1, step=1)

# Only proceed if text and timestamps are provided
if input_text and num_time_stamps:
    tokenizer_path = "/vast/palmer/scratch/liu_xiaofeng/ss4786/venv/model_weights"
    model_path = "/vast/palmer/scratch/liu_xiaofeng/ss4786/venv/model_weights"
    patchtst_path = "/vast/palmer/scratch/liu_xiaofeng/ss4786/sourav/alpha_beta_variation_ckpts/model_best_patchtst_7_6.pt"
    classifier_path = "/vast/palmer/scratch/liu_xiaofeng/ss4786/sourav/alpha_beta_variation_ckpts/model_best_classifier_7_6.pt"
    max_len = 4094
    alpha = 0.7 # weight for classifier output
    beta = 0.6  # weight for PatchTST output
    threshold = 0.566  # threshold for anorexia prediction

    # Load models
    tokenizer, longformer_model = load_longformer_model(tokenizer_path, model_path)

    if tokenizer and longformer_model:
        # Preprocess the text
        input_ids, attn_mask = preprocess_text(input_text, tokenizer, max_len)

        if input_ids is not None and attn_mask is not None:
            # Get embeddings from Longformer
            longformer_embeddings = get_longformer_embeddings(longformer_model, input_ids, attn_mask)
            
            if longformer_embeddings is not None:
                # Collect the time stamps data
                padded_time, past_observed_mask = get_time_stamps(num_time_stamps)

                if padded_time is not None and past_observed_mask is not None:
                    # Load models for PatchTST and Classifier
                    model_patchtst = PatchTSTForClassification(config=config)
                    model_classifier = BinaryClassifier()
                    
                    model_patchtst = load_model_weights(model_patchtst, patchtst_path)
                    model_classifier = load_model_weights(model_classifier, classifier_path)
                    
                    if model_patchtst and model_classifier:
                        # Make the predictions
                        output1 = model_patchtst(past_values=padded_time.to(Device), 
                                                 past_observed_mask=past_observed_mask.to(Device))
                        output1 = sigmoid(output1.prediction_logits)

                        output2 = model_classifier(longformer_embeddings.to(Device))

                        # Combine the outputs
                        output_combined = alpha * output2 + beta * output1
                        output_combined = torch.clamp(output_combined, max=1).item()

                        # Display the result
                        if output_combined > threshold:
                            st.success("The patient has anorexia!")
                        else:
                            st.success("The patient does not have anorexia!")
