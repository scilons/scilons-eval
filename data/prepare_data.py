import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict
from data.data_prep_utils import (
    read_txt_file_ner,
    read_txt_file_pico,
    tokenize_data_ner,
    prepare_input_ner,
    extract_labels,
    get_labels_rel_cls,
    extract_texts_labels_rel_cls,
    tokenize_function,
    prepare_input_rel_cls,
)
from typing import Dict
import os

class DatasetPrep:
    def __init__(self, task, data_path, tokenizer, device) -> None:
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

    def run(self) -> (DatasetDict, Dict):
        """
        A function that returns a DatasetDict object that includes train, dev, and test data samples. 
        Additionally the function returns a dictionary of all labels in the dataset mapped to categorical numbers. 
        """

        data_types = ["train", "dev", "test"]

        data_paths = [self.data_path + "/" + dataset + ".txt" for dataset in data_types]
        
        if self.task == "ner" or self.task == "pico":
            labels_set = self._get_labels_ner_pico(data_paths)
        elif self.task == "rel" or self.task == "cls":
            labels_set = get_labels_rel_cls(data_paths)

        labels_mapper = {label: i for i, label in enumerate(labels_set)}

        dataset_dict = DatasetDict()

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
                tokenized_sentences = tokenize_function(sentences, self.tokenizer)

            # Get tokenized input based on task
            if self.task == "ner" or self.task == "pico":
                token_ids, attention_masks, token_type_ids, labels = prepare_input_ner(
                    tokenized_sentences, self.tokenizer, labels_mapper, self.device
                )
            elif self.task == "rel" or self.task == "cls":
                token_ids, attention_masks, token_type_ids, labels = prepare_input_rel_cls(
                    tokenized_sentences, labels_mapper, categor_labels, self.device
                )

            # Create a Dataset object
            dataset_inputs = Dataset.from_dict(
                {
                    "input_ids": token_ids,
                    "attention_mask": attention_masks,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                }
            )

            directory = path.split(os.path.sep)
            dataset_name = directory[-1]

            dataset_dict[dataset_name[:-4]] = dataset_inputs

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
