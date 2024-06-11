import torch
from torch.nn.utils.rnn import pad_sequence
import os
import json
from typing import List, Tuple


def read_txt_file_ner(file_path):
    directories = file_path.split(os.path.sep)
    dataset_name = directories[-2]

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    sentences = []
    sentence_tokens = []
    for line in lines:
        if line.strip() == "" or line.startswith("-DOCSTART-"):
            if sentence_tokens:
                sentences.append(sentence_tokens)
                sentence_tokens = []
        else:
            if dataset_name == "sciie":
                token, pos_tag, _, ner_tag = line.strip().split(" ")
                sentence_tokens.append((token, ner_tag))
            else:
                token, pos_tag, _, ner_tag = line.strip().split("\t")
                sentence_tokens.append((token, ner_tag))
    if sentence_tokens:  # Append the last sentence if not empty
        sentences.append(sentence_tokens)

    return sentences


def read_txt_file_pico(file_path):
    data = []  # List to store parsed data
    current_sentence = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            elif line == "":
                if current_sentence:
                    data.append(current_sentence)
                    current_sentence = []
                continue
            else:
                tokens = line.split()
                word = tokens[0]
                label = tokens[-1]
                current_sentence.append((word, label))

    return data


def tokenize_data_ner(sentences, tokenizer):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = []
        labels = []
        for token, label in sentence:
            tokenized_token = tokenizer.tokenize(token)
            tokens.extend(tokenized_token)
            labels.extend([label] * len(tokenized_token))
        tokenized_sentences.append((tokens, labels))
    return tokenized_sentences

def extract_labels(tokenized_sentences):
    all_labels = []
    for tokens, labels in tokenized_sentences:
        all_labels.extend(labels)
    return all_labels


# Function to extract spans from labels for evaluating span-level metrics
def extract_spans(labels, labels_mapper):
    spans = []
    current_span = None
    label_map_inv = {v: k for k, v in labels_mapper.items()}

    for i, label_id in enumerate(labels):
        label = label_map_inv[label_id]

        if label.startswith("B-"):
            if current_span is not None:
                spans.append(current_span)
            current_span = {"type": label[2:], "start": i, "end": i}

        elif label.startswith("I-"):
            if current_span is not None:
                if current_span["type"] == label[2:]:
                    current_span["end"] = i
                else:
                    spans.append(current_span)
                    current_span = {"type": label[2:], "start": i, "end": i}
            else:
                current_span = {"type": label[2:], "start": i, "end": i}

        else:  # Outside tag
            if current_span is not None:
                spans.append(current_span)
                current_span = None

    if current_span is not None:
        spans.append(current_span)

    return spans


def get_labels_rel_cls(file_paths: List[str]) -> set:
    
    labels = set()

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    labels.add(data["label"])
                except json.JSONDecodeError as error:
                    print(f"Error decoding JSON from line: {line}")
                    print(error)
    return labels


def extract_texts_labels_rel_cls(file_path: str) -> Tuple:

    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                texts.append(data["text"])
                labels.append(data["label"])

            except json.JSONDecodeError as error:
                print(f"Error decoding JSON from line: {line}")
                print(error)

    return texts, labels


def prepare_input_rel_cls(tokenized_sentences, labels_mapper: dict, labels: list, device) -> Tuple:
    """
    Prepares tokenized sentences and encodes labels.
    Args:
        tokenized_sentences: Input ids, attention mask, and token type ids.
        labels_mapper (dict): A dictionary with labels as keys and the corresponding numbers as values.
        labels (list): List of labels to encode.
        device: Torch device.
    Returns:
        Tuple of token ids, attention masks, token type ids, and encoded labels.
    """

    token_ids = tokenized_sentences["input_ids"].to(device)
    attention_masks = tokenized_sentences["attention_mask"].to(device)
    token_type_ids = tokenized_sentences["token_type_ids"].to(device)
    labels_encoded = [labels_mapper[label] for label in labels]

    return token_ids, attention_masks, token_type_ids, labels_encoded

def extract_dep_data(file_path: str) -> List[Tuple[str, str, str]]:
    
    with open(file_path, "r", encoding="utf-8") as file:
        
        lines = file.readlines()
        sentences = []
        sentence_tokens = []
        
        for line in lines:
            if not line.strip():
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    sentence_tokens = []
            else:
                idx, token, lemma, upos, xpos, _, head, dep_tag, _, _ =  line.strip().split("\t")
                sentence_tokens.append((token, dep_tag, head))
        if sentence_tokens:  
            sentences.append(sentence_tokens)

    return sentences 

def get_labels_dep(file_paths: List[str]) -> set:
        
    labels_set = set()
     
    for file_path in file_paths: 
        sentences = extract_dep_data(file_path)
        labels = [element[1] for item in sentences for element in item]
        for item in labels:
            labels_set.add(item)

    return labels_set

def tokenize_data_dep(sentences: List, tokenizer) -> List:
    
    tokenized_sentences = []
    
    for sentence in sentences:
        
        tokens = []
        labels = []
        
        for token, label, head in sentence:
            tokenized_token = tokenizer.tokenize(token)
            tokens.extend(tokenized_token)
            labels.extend([label] * len(tokenized_token))
  
        tokenized_sentences.append((tokens, labels))
    return tokenized_sentences

def prepare_input_dep(tokenized_data: List, 
                      tokenizer, 
                      label_map: set, 
                      device) -> Tuple:
    
    token_ids = []
    attention_masks = []
    token_type_ids = []
    encoded_labels = []

    max_seq_length = max(len(tokens) for tokens, _ in tokenized_data)
    
    for tokens, labels in tokenized_data:
        joined_text = " ".join(tokens)
        encoded_dict = tokenizer.encode_plus(joined_text,
                                             add_special_tokens=True,
                                             max_length=max_seq_length,
                                             padding='max_length',
                                             truncation=True, 
                                             return_attention_mask=True,
                                             return_token_type_ids=True,
                                             return_tensors='pt')
        
        token_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
        token_type_ids.append(encoded_dict["token_type_ids"])
        mapped_labels = [label_map[label] for label in labels]
        encoded_labels.append(torch.tensor(mapped_labels))

    labels_padded = pad_sequence(encoded_labels, 
                                 batch_first=True, 
                                 padding_value = -100)
    labels_padded.to(device)
    token_ids = torch.cat(token_ids, dim=0)
    token_ids.to(device)
    attention_masks = torch.cat(attention_masks, dim=0)
    attention_masks.to(device)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    token_type_ids.to(device)

    
    return token_ids, attention_masks, token_type_ids, labels_padded


def get_labels_heads(file_paths: List[str]) -> set:
    
    labels_set = set()
     
    for file_path in file_paths: 
        sentences = extract_dep_data(file_path)
        labels = [element[2] for item in sentences for element in item]
        for item in labels:
            labels_set.add(item)

    return labels_set


def tokenize_data_heads(sentences: List, tokenizer) -> list:
    
    tokenized_sentences = []
    
    for sentence in sentences:
        
        tokens = []
        labels = []
        
        for token, label, head in sentence:
            tokenized_token = tokenizer.tokenize(token)
            tokens.extend(tokenized_token)
            labels.extend([head] * len(tokenized_token))
        tokenized_sentences.append((tokens, labels))

    return tokenized_sentences

