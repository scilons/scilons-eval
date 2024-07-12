import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict
from data.data_prep_utils import (
    read_txt_file_ner,
    read_txt_file_pico,
    tokenize_data_ner,
    extract_labels,
    get_labels_rel_cls,
    extract_texts_labels_rel_cls,
    prepare_input_rel_cls,
    extract_dep_data,
    get_labels_dep,
    tokenize_data_dep,
    get_labels_heads,
    tokenize_data_heads
)
from typing import Dict, List, Tuple
import os

class DatasetPrep:
    def __init__(self, task, data_path, tokenizer, device, max_length) -> None:
        """
        task: should be 'ner', ....
        data_path: str that describes the path where the train.txt, dev.txt, and test.txt datasets are
        tokenizer: Transformer's tokenizer according to the model used
        device: current device
        """
        self.task = task
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def run(self) -> (DatasetDict, Dict):
        """
        A function that returns a DatasetDict object that includes train, dev, and test data samples. 
        Additionally the function returns a dictionary of all labels in the dataset mapped to categorical numbers. 
        """

        data_types = ["train", "dev", "test"]

        data_paths = [self.data_path + "/" + dataset + ".txt" for dataset in data_types]

        if self.task == "dep":
            labels_set_dep = get_labels_dep(data_paths)
            labels_set_head = get_labels_heads(data_paths)
            labels_mapper_dep = {label: i for i, label in enumerate(labels_set_dep)} 
            labels_mapper_head = {label: i for i, label in enumerate(labels_set_head)}
        else:
            if self.task == "ner" or self.task == "pico":
                labels_set = self._get_labels_ner_pico(data_paths)
            elif self.task == "rel" or self.task == "cls":
                labels_set = get_labels_rel_cls(data_paths)
                
            labels_mapper = {label: i for i, label in enumerate(labels_set)}
            
        dataset_dict = DatasetDict()
        dataset_dict_dep = DatasetDict()
        dataset_dict_heads = DatasetDict()

        for path in data_paths:
            # Read txt file based on given task
            if self.task == "ner":
                sentences = read_txt_file_ner(path)
                tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)

            elif self.task == "pico":
                sentences = read_txt_file_pico(path)
                tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)

            elif self.task == "rel" or self.task == "cls":
                sentences, categor_labels = extract_texts_labels_rel_cls(path)
                tokenized_sentences = self.tokenize_function(sentences)

            elif self.task == "dep":
                sentences = extract_dep_data(path)
                tokenized_sentences_dep = tokenize_data_dep(sentences, self.tokenizer)
                tokenized_sentences_heads = tokenize_data_heads(sentences, self.tokenizer)

            # Get tokenized input based on task
            if self.task == "ner" or self.task == "pico":
                token_ids, attention_masks, token_type_ids, labels = self.prepare_input_ner(
                    tokenized_sentences, labels_mapper
                )
            elif self.task == "rel" or self.task == "cls":
                token_ids, attention_masks, token_type_ids, labels = prepare_input_rel_cls(
                    tokenized_sentences, labels_mapper, categor_labels, self.device
                )
            elif self.task == "dep":
                token_ids_dep, attention_masks_dep, token_type_ids_dep, labels_dep = self.prepare_input_dep(
                    tokenized_sentences_dep, self.tokenizer, labels_mapper_dep, self.device
                )                      
                
                token_ids_heads, attention_masks_heads, token_type_ids_heads, labels_heads = self.prepare_input_dep(
                    tokenized_sentences_heads, self.tokenizer, labels_mapper_head, self.device
                )

            directory = path.split(os.path.sep)
            dataset_name = directory[-1]

            if self.task == "dep":
                
                dataset_inputs_dep = Dataset.from_dict(
                    {
                        "input_ids": token_ids_dep,
                        "attention_mask": attention_masks_dep,
                        "token_type_ids": token_type_ids_dep,
                        "labels": labels_dep,
                    }
                )
                dataset_dict_dep[dataset_name[:-4]] = dataset_inputs_dep
                
                dataset_inputs_heads = Dataset.from_dict(
                    {
                        "input_ids": token_ids_heads,
                        "attention_mask": attention_masks_heads,
                        "token_type_ids": token_type_ids_heads,
                        "labels": labels_heads,
                    }
                )

                dataset_dict_heads[dataset_name[:-4]] = dataset_inputs_heads

            else:
                
                dataset_inputs = Dataset.from_dict(
                    {
                        "input_ids": token_ids,
                        "attention_mask": attention_masks,
                        "token_type_ids": token_type_ids,
                        "labels": labels
                    }
                )
                dataset_dict[dataset_name[:-4]] = dataset_inputs

        if self.task == "dep":
            return dataset_dict_dep, dataset_dict_heads, labels_mapper_dep, labels_mapper_head
        
        return dataset_dict, labels_mapper

    def _get_labels_ner_pico(self, file_paths):
    
        if self.task == "ner":
            sentences = [read_txt_file_ner(file) for file in file_paths]
        elif self.task == "pico":
            sentences = [read_txt_file_pico(file) for file in file_paths]

        sentences = [item for row in sentences for item in row]

        tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
        all_labels = extract_labels(tokenized_sentences)
        labels_list = list(set(all_labels))

        return labels_list

    def tokenize_function(self, texts: list):
        
        return self.tokenizer(
            texts, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt", 
            padding=True,
            return_token_type_ids=True
        )

    def prepare_input_dep(self, tokenized_data: List, tokenizer, label_map: set, device) -> Tuple:
        
        token_ids = []
        attention_masks = []
        token_type_ids = []
        encoded_labels = []
        
        for tokens, labels in tokenized_data:
            joined_text = " ".join(tokens)
            encoded_dict = tokenizer.encode_plus(joined_text,
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation=True, 
                                                 return_attention_mask=True,
                                                 return_token_type_ids=True,
                                                 return_tensors='pt')
            token_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])
            mapped_labels = [label_map[label] for label in labels]
            encoded_labels.append(torch.tensor(mapped_labels[:self.max_length]))
            
        token_ids = torch.cat(token_ids, dim=0)
        token_ids.to(device)
        attention_masks = torch.cat(attention_masks, dim=0)
        attention_masks.to(device)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        token_type_ids.to(device)
        
        return token_ids, attention_masks, token_type_ids, encoded_labels

    def prepare_input_ner(self, tokenized_data, label_map):
        
        token_ids = []
        attention_masks = []
        token_type_ids = []
        labels = []
        
        for tokens, entity_labels in tokenized_data:
            joined_text = " ".join(tokens)
            encoded_dict = self.tokenizer.encode_plus(joined_text,
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation=True, 
                                                 return_attention_mask=True,
                                                 return_token_type_ids=True,
                                                 return_tensors='pt')
            token_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])
            labels.append(torch.tensor([label_map[label] for label in entity_labels][:self.max_length]))
            
        token_ids = torch.cat(token_ids, dim=0)
        token_ids.to(self.device)
        attention_masks = torch.cat(attention_masks, dim=0)
        attention_masks.to(self.device)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        token_type_ids.to(self.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=label_map["O"])
        labels.to(self.device)
        
        return token_ids, attention_masks, token_type_ids, labels
