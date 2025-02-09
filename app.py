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
from utils import *

# Initialize necessary components
sigmoid = nn.Sigmoid()
Device = "cuda" if torch.cuda.is_available() else "cpu"

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
