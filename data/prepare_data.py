import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from data.data_prep_utils import read_txt_file_ner, tokenize_data_ner, prepare_input_ner, extract_labels


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
        
        if self.task == "ner":
            dataset_dict, labels_mapper = self.prep_ner_data()
            return dataset_dict, labels_mapper
        else:
            return None

    def prep_ner_data(self):

        data_types = ['train', 'dev', 'test']
        labels_list = self._get_labels_list()
        labels_mapper = {label: i for i, label in enumerate(set(labels_list))}
        dataset_dict = DatasetDict()

        for dataset in data_types:
            sentences = read_txt_file_ner(self.data_path + "/" + dataset + ".txt")
            tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
            token_ids, attention_masks, token_type_ids, labels = prepare_input_ner(tokenized_sentences, self.tokenizer,labels_mapper, self.device)
            
            dataset_inputs = Dataset.from_dict({
                'input_ids': token_ids,
                'attention_mask': attention_masks,
                'token_type_ids': token_type_ids,
                'labels': labels
                })

            dataset_dict[dataset] = dataset_inputs

        return dataset_dict, labels_mapper


    def _get_labels_list(self):
        sentences = read_txt_file_ner(self.data_path + "/train.txt")
        tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
        all_labels = extract_labels(tokenized_sentences)
        labels_list = list(set(all_labels))

        return labels_list
        