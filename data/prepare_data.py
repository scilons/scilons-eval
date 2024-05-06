import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from datasets import Dataset, DatasetDict
from data.data_prep_utils import (
    read_txt_file_ner,
    read_txt_file_pico,
    tokenize_data_ner,
    prepare_input_ner,
    extract_labels,
    collect_labels_rel_cls,
    extract_texts_labels_rel_cls,
    tokenize_function,
    prepare_input_rel_cls,
)


class DatasetPrep:
    def __init__(self, task, data_path, tokenizer, device) -> None:
        """
        task: should be 'ner', ....
        data_path: str that describes the path where the train.txt, dev.txt, and test.txt datasets are
        tokenizer: Transformer's tokenizer according to the model used
        device: current device
        label: list of labels used in the current task
        """
        self.task = task
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.device = device

    def run(self) -> Dataset:
        dataset_dict, labels_mapper = self.prep_data()
        return dataset_dict, labels_mapper

    def prep_data(self):
        data_types = ["train", "dev", "test"]

        if self.task == "ner" or self.task == "pico":
            labels_list = self._get_labels_list()
            labels_mapper = {label: i for i, label in enumerate(set(labels_list))}
        elif self.task == "rel" or self.task == "cls":
            file_paths = [dataset + ".txt" for dataset in data_types]
            file_paths = [os.path.join(self.data_path, path) for path in file_paths]
            labels_set = collect_labels_rel_cls(file_paths)
            labels_mapper = {label: i for i, label in enumerate(labels_set)}

        dataset_dict = DatasetDict()

        for dataset in data_types:
            if self.task == "ner":
                sentences = read_txt_file_ner(self.data_path + "/" + dataset + ".txt")
                tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
            elif self.task == "pico":
                sentences = read_txt_file_pico(self.data_path + "/" + dataset + ".txt")
                tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
            elif self.task == "rel" or self.task == "cls":
                sentences, categor_labels = extract_texts_labels_rel_cls(
                    self.data_path + "/" + dataset + ".txt"
                )
                tokenized_sentences = tokenize_function(sentences, self.tokenizer)

            if self.task == "ner" or self.task == "pico":
                token_ids, attention_masks, token_type_ids, labels = prepare_input_ner(
                    tokenized_sentences, self.tokenizer, labels_mapper, self.device
                )
            elif self.task == "rel" or self.task == "cls":
                (
                    token_ids,
                    attention_masks,
                    token_type_ids,
                    labels,
                ) = prepare_input_rel_cls(
                    tokenized_sentences, labels_mapper, categor_labels, self.device
                )

            dataset_inputs = Dataset.from_dict(
                {
                    "input_ids": token_ids,
                    "attention_mask": attention_masks,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                }
            )

            dataset_dict[dataset] = dataset_inputs

        return dataset_dict, labels_mapper

    def _get_labels_list(self):
        if self.task == "ner":
            sentences = read_txt_file_ner(self.data_path + "/train.txt")
        elif self.task == "pico":
            sentences = read_txt_file_pico(self.data_path + "/train.txt")

        tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
        all_labels = extract_labels(tokenized_sentences)
        labels_list = list(set(all_labels))

        return labels_list
