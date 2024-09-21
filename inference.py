import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import numpy as np
import argparse
from transformers import LongformerTokenizer, LongformerModel
from transformers import PatchTSTConfig, PatchTSTForClassification
import torch.nn as nn
from preprocessing import *
from model import *
from utils import *

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
    # loss='bce',
    head_dropout=0.5,
)

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
        print(f"Error in generating Longformer embeddings: {str(e)}")
        return None

def get_time_stamps(num_time_stamps):
    """Collect and pad time series data."""
    try:
        time = []
        for _ in range(num_time_stamps):
            start = int(input("Enter start time: "))
            end = int(input("Enter end time: "))
            time.append([start, end])
        
        padded_time = np.pad(time, ((0, 2000 - len(time)), (0, 0)), 'constant')
        past_observed_mask = np.ones(2000)
        past_observed_mask[len(time):] = 0
        past_observed_mask = torch.tensor(np.stack((past_observed_mask, past_observed_mask), axis=1), dtype=torch.bool)
        return torch.tensor(padded_time, dtype=torch.float32).unsqueeze(0), past_observed_mask.unsqueeze(0)
    except Exception as e:
        print(f"Error in getting time stamps: {str(e)}")
        return None, None

def main(args):
    try:
        # Load Longformer model and tokenizer
        tokenizer, longformer_model = load_longformer_model(args.tokenizer_path, args.model_path)
        
        if tokenizer is None or longformer_model is None:
            raise ValueError("Failed to load Longformer components.")
        
        # Input text and get Longformer embeddings
        input_text = args.input_text
        input_ids, attn_mask = preprocess_text(input_text, tokenizer, args.max_len)
        
        if input_ids is None or attn_mask is None:
            raise ValueError("Failed to preprocess input text.")
        
        final_logits = get_longformer_embeddings(longformer_model, input_ids, attn_mask)
        
        if final_logits is None:
            raise ValueError("Failed to generate embeddings from Longformer.")
        
        # Get time series data
        padded_time, past_observed_mask = get_time_stamps(args.num_time_stamps)
        if padded_time is None or past_observed_mask is None:
            raise ValueError("Failed to get time series data.")
        
        # Load PatchTST and classifier models
        model_patchtst = PatchTSTForClassification(config)
        model_classifier = BinaryClassifier()
        
        model_patchtst = load_model_weights(model_patchtst, args.patchtst_path)
        model_classifier = load_model_weights(model_classifier, args.classifier_path)
        
        if model_patchtst is None or model_classifier is None:
            raise ValueError("Failed to load models.")
        
        # Forward pass through models
        output1 = model_patchtst(past_values = padded_time.to(Device), past_observed_mask = past_observed_mask.to(Device))
        output1 = sigmoid(output1.prediction_logits)
        output2 = model_classifier(final_logits.to(Device))
        # Combine model outputs and make prediction
        alpha, beta = args.alpha, args.beta
        output_combined = alpha * output2 + beta * output1
        output_combined = torch.clamp(output_combined, max=1).item()
        # Final prediction
        if output_combined > args.threshold:
            print("The patient has anorexia!")
        else:
            print("The patient does not have anorexia!")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Argument parser for configurable parameters
    parser = argparse.ArgumentParser(description="Anorexia Prediction using Longformer and PatchTST")
    
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to Longformer tokenizer')
    parser.add_argument('--model_path', type=str, default=None, help='Path to Longformer model')
    parser.add_argument('--patchtst_path', type=str, default="/vast/palmer/scratch/liu_xiaofeng/ss4786/sourav/alpha_beta_variation_ckpts/model_best_patchtst_7_6.pt", help='Path to PatchTST model')
    parser.add_argument('--classifier_path', type=str, default="/vast/palmer/scratch/liu_xiaofeng/ss4786/sourav/alpha_beta_variation_ckpts/model_best_classifier_7_6.pt", help='Path to Binary Classifier model')
    
    parser.add_argument('--input_text', type=str, default="hello world", help='Input text for the model')
    parser.add_argument('--max_len', type=int, default=4094, help='Maximum length of tokens for Longformer')
    
    parser.add_argument('--num_time_stamps', type=int, default=1, help='Number of time stamps for the time series data')
    parser.add_argument('--alpha', type=float, default=0.9, help='Weight for classifier output')
    parser.add_argument('--beta', type=float, default=0.2, help='Weight for PatchTST output')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for the final prediction')

    args = parser.parse_args()
    
    main(args)
