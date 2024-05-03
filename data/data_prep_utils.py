import torch
from torch.nn.utils.rnn import pad_sequence
import os

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
            if dataset_name == 'sciie':
                token, pos_tag, _, ner_tag = line.strip().split(" ")
                sentence_tokens.append((token, ner_tag))
            else:
                token, pos_tag, _, ner_tag = line.strip().split("\t")
                sentence_tokens.append((token, ner_tag))
    if sentence_tokens:  # Append the last sentence if not empty
        sentences.append(sentence_tokens)
        
    return sentences

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

def prepare_input_ner(tokenized_data, tokenizer, label_map, device):
    
    token_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    max_seq_length = max(len(tokens) for tokens, _ in tokenized_data)

    for tokens, entity_labels in tokenized_data:
        encoded_dict = tokenizer.encode_plus(tokens, max_length=max_seq_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
        token_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        labels.append(torch.tensor([label_map[label] for label in entity_labels]))
    

    token_ids = torch.cat(token_ids, dim=0)
    token_ids.to(device)
    attention_masks = torch.cat(attention_masks, dim=0)
    attention_masks.to(device)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    token_type_ids.to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=label_map['O'])
    labels.to(device)

    return token_ids, attention_masks, token_type_ids, labels

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

