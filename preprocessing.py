from torch import Tensor
import torch
from transformers import PreTrainedTokenizerBase
import re
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerModel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as AMFSC
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import random
from datetime import datetime
import nltk


nltk.data.path.append('/home/ajeet/Sourav/pre_trained_weigths')
nltk.download('punkt', 
              download_dir='/home/ajeet/Sourav/pre_trained_weigths')
nltk.download('averaged_perceptron_tagger', 
              download_dir='/home/ajeet/Sourav/pre_trained_weigths')
nltk.download('wordnet', 
              download_dir='/home/ajeet/Sourav/pre_trained_weigths')
nltk.download('omw-1.4', 
              download_dir='/home/ajeet/Sourav/pre_trained_weigths')


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


def extract_lemmas(text):
    keep_terms = ["why", "what", "how", "where", "when", "which"]
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    lemmas = []
    for word, tag in tagged_tokens:
        if word.lower() in keep_terms:  # Directly add specific terms to the output
            lemmas.append(word.lower())
            continue
        wn_tag = get_wordnet_pos(tag)
        if wn_tag in (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV):  # Only proceed if noun or verb
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            lemmas.append(lemma)
        elif wn_tag is None and tag.startswith('NN'):  # Fallback for nouns without a clear WN tag
            lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
            lemmas.append(lemma)
    
    return lemmas


def get_wordnet_pos(treebank_tag):
    """Converts treebank tags to wordnet tags."""
    # if treebank_tag.startswith('J'):
    #     return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    # elif treebank_tag.startswith('R'):
    #     return wordnet.ADV
    else:
        return None


def get_synonyms(word, pos=None):
    """Fetches synonyms for a word based on its part of speech."""
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def augment_sentence(sentence, num_augmentations=1):
    """Augments a sentence by replacing words with their synonyms."""
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    
    augmented_sentences = [sentence]
    for _ in range(num_augmentations):
        new_sentence = words.copy()
        for i, (word, tag) in enumerate(pos_tags):
            wn_tag = get_wordnet_pos(tag)
            if wn_tag:
                synonyms = get_synonyms(word, pos=wn_tag)
                if synonyms:
                    synonym = random.choice(synonyms)
                    new_sentence[i] = synonym
        augmented_sentences.append(' '.join(new_sentence))
    
    return augmented_sentences


def parse_xml_sentiment(file_path, label, device='cpu'):
    tree = ET.parse(file_path)
    root = tree.getroot()
    model_name = "blanchefort/rubert-base-cased-sentiment-med"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            cache_dir="/media/souravsaini/Data/POP_OS/internship/env/model_weights")
    model = AMFSC.from_pretrained(model_name,
                                cache_dir="/media/souravsaini/Data/POP_OS/internship/env/model_weights")

    # Extract ID
    id = root.find('ID').text.strip()

    # Extract writings
    writings = root.findall('WRITING')

    # Initialize list to store sentiment scores
    sentiment_scores = []
    time = []

    for writing in writings:
        title = writing.find('TITLE').text.strip()
        text = writing.find('TEXT').text.strip()
        date = writing.find('DATE').text.strip()

        date = date.split(" ")

        # Extract the time part (hour, minute, second)
        time_parts = date[1].split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        time.append([hour, minute])

        # Concatenate TITLE and TEXT
        concatenated_text = clean_text(f"{title} {text}")

        try:
            # Calculate sentiment score using BERT
            tokens = transform_single_text(concatenated_text, tokenizer, 510, 510, 1)
            model.eval()
            with torch.no_grad():
                out = model(tokens[0].to(device), tokens[1].to(device))
                out = F.softmax(out.logits, dim = -1)

            # Append the sentiment score to the list
            sentiment_scores.append(out[0][2].cpu().item())
        except:
            sentiment_scores.append(0.5)

    # Create the dictionary
    result_dict = {
        'time': time,
        'sentiment_scores': sentiment_scores,
        'label': label  # Assuming this is for the negative class
    }

    return id, result_dict


def parse_xml_longformer_patchtst(text, id, save_dir, model, tokenizer, device='cpu'):
    Device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(Device)

    try:
        temp = []
        tokens = transform_single_text(text, tokenizer, 4000, 4000, 1)
        model.eval()
        for input_id, mask in zip(tokens[0], tokens[1]):
            with torch.no_grad():
                output = model(input_id.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
            temp.append(output.last_hidden_state.squeeze().cpu())
        final_logits = torch.cat(temp, dim=0)
        logits_path = os.path.join(save_dir, f"{id}.npy")
        np.save(logits_path, final_logits)

    except: 
        print(f"id = {id} failed to get logits")
        logits_path = 'dummy'