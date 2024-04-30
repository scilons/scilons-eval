import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from data_prep_utils import read_txt_file_ner, tokenize_data_ner, prepare_input_ner, extract_labels


class DatasetPrep:

    def __init__(self, task, data_path, tokenizer, device, labels_list) -> None:
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
        self.labels = labels_list
    
    def run(self) -> Dataset:
        
        if self.task == "ner":
            dataset_dict = self.prep_ner_data()
        else:
            return None

    def prep_ner_data(self):

        date_types = ['train', 'dev', 'test']
        labels_mapper = {label: i for i, label in enumerate(set(all_labels)}
        num_labels = len(label_map)
        dataset_dict = {}

        for dataset in data_types:
            sentences = read_txt_file_ner(self.data_path + "/" + dataset + ".txt")
            tokenized_sentences = tokenize_data_ner(sentences, self.tokenizer)
            token_ids, attention_masks, token_type_ids, labels = prepare_input_ner(train_tokenized, self.tokenizer,labels_mapper, self.device)
            
            dataset = Dataset.from_dict({
                'input_ids': train_token_ids,
                'attention_mask': attention_masks,
                'token_type_ids': token_type_ids,
                'labels': labels
                })

            dataset_dict[dataset] = dataset

        return dataset_dict



