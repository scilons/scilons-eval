from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForTokenClassification
)
from data.prepare_data import DatasetPrep
from data.data_prep_utils import extract_spans, extract_dep_data
from dep_eval_utils import calculate_uas_las, collect_predictions_dep, collect_gold_labels_dep
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import argparse
import torch
import os
from dataclasses import dataclass, field
from typing import Optional


class ModelEval:
    def __init__(self, task, model_name, tokenizer_name, data_path, device, hf_args, hf_token, max_length):
        self.task = task
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.device = device
        self.hf_args = hf_args
        self.hf_token = hf_token
        self.max_length = max_length

    def train_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, token=self.hf_token, trust_remote_code=True)
        
        dataset_prep = DatasetPrep(
            task=self.task,
            data_path=self.data_path,
            tokenizer=tokenizer,
            device=self.device,
            max_length = self.max_length
        )
        
        if self.task == "dep":
            dataset_dict_dep, dataset_dict_heads, labels_mapper_dep, labels_mapper_head = dataset_prep.run()

        else:
            dataset_dict, labels_mapper = dataset_prep.run()

        if self.task == "ner" or self.task == "pico":
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper), token=self.hf_token, trust_remote_code=True
            )
            data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=labels_mapper["O"])
        elif self.task == "rel" or self.task == "cls":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper), token=self.hf_token, trust_remote_code=True
            )
        elif self.task == "dep":
            model_dep = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper_dep), token=self.hf_token, trust_remote_code=True
            )
            model_heads = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper_head), token=self.hf_token, trust_remote_code=True
            )
            data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100, padding='max_length', max_length=self.max_length)

        if self.task == "dep":
            trainer_dep = Trainer(
                model=model_dep,
                args=self.hf_args,
                train_dataset=dataset_dict_dep["train"],
                eval_dataset=dataset_dict_dep["dev"],
                data_collator=data_collator
                )
            
            trainer_heads = Trainer(
                model=model_heads,
                args=self.hf_args,
                train_dataset=dataset_dict_heads["train"],
                eval_dataset=dataset_dict_heads["dev"],
                data_collator=data_collator
                )
            
            trainer_dep.train()
            trainer_dep.save_model("results/models")

            trainer_heads.train()
            trainer_heads.save_model("results/models")

            return model_dep, model_heads, dataset_dict_dep, dataset_dict_heads, labels_mapper_dep, labels_mapper_head

        if self.task == "ner" or self.task == "pico":
            
            trainer = Trainer(
                model=model,
                args=self.hf_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["dev"],
                data_collator=data_collator
            )

            trainer.train()
            trainer.save_model("results/models")
            
            return model, dataset_dict, labels_mapper
            
            
        trainer = Trainer(
            model=model,
            args=self.hf_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["dev"],
        )

        trainer.train()

        trainer.save_model("results/models")

        return model, dataset_dict, labels_mapper

    def evaluate_model(self):
        
        if self.task == "dep":
            
            (
                trained_model_dep, 
                trained_model_heads, 
                dataset_dict_dep, 
                dataset_dict_heads, 
                labels_mapper_dep, 
                labels_mapper_head
            ) = self.train_model()
            
            trained_model_dep.eval()

            reverse_label_map_dep = {v: k for k, v in labels_mapper_dep.items()}
            reverse_label_map_head = {v: k for k, v in labels_mapper_head.items()}

            pred_dep_labels = collect_predictions_dep(dataset_dict_dep["test"], 
                                                      trained_model_dep, 
                                                      reverse_label_map_dep,
                                                      self.device)

            gold_dep_labels = collect_gold_labels_dep(dataset_dict_dep["test"]["labels"],
                                                      reverse_label_map_dep)

            pred_head_labels = collect_predictions_dep(dataset_dict_heads["test"], 
                                                       trained_model_heads, 
                                                       reverse_label_map_head ,
                                                       self.device)
            
            gold_head_labels =  collect_gold_labels_dep(dataset_dict_heads["test"]["labels"],
                                                        reverse_label_map_head)
            
            words_test_set = extract_dep_data(self.data_path + "/test.txt")
            words_test_set = [item[0] for sample in words_test_set for item in sample]

            uas, las = calculate_uas_las(words_test_set, 
                                         gold_head_labels, 
                                         gold_dep_labels, 
                                         pred_head_labels, 
                                         pred_dep_labels)

            return uas, las

        trained_model, dataset_dict, labels_mapper = self.train_model()
        trained_model.eval()

        # NER is evaluated according to span-level macro F1
        if self.task == "ner":
            true_spans_by_type = defaultdict(list)
            predicted_spans_by_type = defaultdict(list)

            for sample in dataset_dict["test"]:
                inputs = {
                    key: torch.tensor(sample[key]).unsqueeze(0).to(self.device)
                    for key in sample.keys()
                    if key != "labels"
                }
                labels = sample["labels"]

                with torch.no_grad():
                    outputs = trained_model(**inputs)
                    predictions = (
                        torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
                    )

                true_spans = extract_spans(labels, labels_mapper)
                predicted_spans = extract_spans(predictions, labels_mapper)

                for span in true_spans:
                    true_spans_by_type[span["type"]].append(span)

                for span in predicted_spans:
                    predicted_spans_by_type[span["type"]].append(span)

            # Compute F1 score for each span type
            span_types = set(
                list(true_spans_by_type.keys()) + list(predicted_spans_by_type.keys())
            )
            macro_f1_scores = []

            for span_type in span_types:
                true_positives = sum(
                    1
                    for span in predicted_spans_by_type[span_type]
                    if span in true_spans_by_type[span_type]
                )
                false_positives = sum(
                    1
                    for span in predicted_spans_by_type[span_type]
                    if span not in true_spans_by_type[span_type]
                )
                false_negatives = sum(
                    1
                    for span in true_spans_by_type[span_type]
                    if span not in predicted_spans_by_type[span_type]
                )

                precision = true_positives / (true_positives + false_positives + 1e-9)
                recall = true_positives / (true_positives + false_negatives + 1e-9)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

                macro_f1_scores.append(f1)

            macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores)
            micro_f1 = None

        # PICO, REL, and CLS are evaluated based on sample-level (either token or sentence) macro-F1
        elif self.task == "pico" or self.task == "rel" or self.task == "cls":
            predictions = []
            true_labels = []

            for sample in dataset_dict["test"]:
                inputs = {
                    key: torch.tensor(sample[key]).unsqueeze(0).to(self.device)
                    for key in sample.keys()
                    if key != "labels"
                }
                labels = sample["labels"]

                with torch.no_grad():
                    outputs = trained_model(**inputs)
                    predicted_labels = (
                        torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
                    )

                if self.task == "pico":
                    # Trim the predicted_labels to match the length of the true labels
                    trimmed_predictions = predicted_labels[:len(labels)]
                    # Results for pico are lists but integers for the other tasks
                    predictions.extend(trimmed_predictions)
                    true_labels.extend(labels)
                else:
                    predictions.append(predicted_labels)
                    true_labels.append(labels)

            micro_f1 = f1_score(true_labels, predictions, average="micro")
            macro_f1 = f1_score(true_labels, predictions, average="macro")

        return micro_f1, macro_f1

@dataclass
class CustomArguments:
    task: str = field(
        metadata={"help": "Specify the task name"}
    )
    model: str = field(
        metadata={"help": "Specify the model name from HuggingFace"}
    )
    tokenizer: str = field(
        metadata={"help": "Specify the tokenizer name from HuggingFace"}
    )
    hf_token: str = field(
        metadata={"help": "Specify your HuggingFace token to access a closed modek."}
    )
    max_length: int = field(
        metadata={"help": "Specify the maximum sequence length of the model to use when tokenizing."}
    )
    data: str = field(
        metadata={"help": "Specify the data path (the folder that contains train.txt, dev.txt, and test.txt)"}
    )

def main():

    parser = HfArgumentParser((TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()

    task = custom_args.task
    model_name = custom_args.model
    tokenizer_name = custom_args.tokenizer
    hf_token = custom_args.hf_token
    max_length = custom_args.max_length
    data_path = custom_args.data

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_eval = ModelEval(
        task=task,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        data_path=data_path,
        device=device,
        hf_args = training_args,
        hf_token = hf_token, 
        max_length = max_length
    )

    if task == 'dep':
        uas, las = model_eval.evaluate_model()
        print("UAS score: ", uas)
        print("LAS score: ", las)
        
    else:
        micro_f1, macro_f1 = model_eval.evaluate_model()

    if task == 'ner':
        print("Macro F1 score (span-level): ", macro_f1)
    elif task == 'pico':
        print("Micro F1 score (token-level): ", micro_f1)
        print("Macro F1 score (token-level): ", macro_f1)
    elif task == 'rel' or task == 'cls':
        print("Micro F1 score (sentence-level): ", micro_f1)
        print("Macro F1 score (sentence-level): ", macro_f1)
        
if __name__ == "__main__":
    main()
