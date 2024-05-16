from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from data.prepare_data import DatasetPrep
from data.data_prep_utils import extract_spans
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import argparse
import torch
import os
from dataclasses import dataclass, field
from typing import Optional


class ModelEval:
    def __init__(self, task, model_name, tokenizer_name, data_path, device, hf_args):
        self.task = task
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.device = device
        self.hf_args = hf_args

    def train_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        dataset_prep = DatasetPrep(
            task=self.task,
            data_path=self.data_path,
            tokenizer=tokenizer,
            device=self.device,
        )
        dataset_dict, labels_mapper = dataset_prep.run()

        if self.task == "ner" or self.task == "pico":
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper)
            )
        elif self.task == "rel" or self.task == "cls":
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(labels_mapper)
            )

        training_args = TrainingArguments(
            output_dir=self.hf_args.output_dir,
            num_train_epochs=self.hf_args.num_train_epochs,
            per_device_train_batch_size=self.hf_args.per_device_train_batch_size,
            report_to=self.hf_args.report_to,
            logging_steps=self.hf_args.logging_steps,
            save_steps=self.hf_args.save_steps,
            evaluation_strategy=self.hf_args.evaluation_strategy,
            eval_steps=self.hf_args.eval_steps,
            learning_rate=self.hf_args.learning_rate
        )

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

            macro_f1_score = sum(macro_f1_scores) / len(macro_f1_scores)

            print("Macro F1 score (span-level): ", macro_f1_score)

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
                    # Results for pico are lists but integers for the other tasks
                    predictions.extend(predicted_labels)
                    true_labels.extend(labels)
                else:
                    predictions.append(predicted_labels)
                    true_labels.append(labels)

            micro_f1 = f1_score(true_labels, predictions, average="micro")
            macro_f1 = f1_score(true_labels, predictions, average="macro")

            print("Micro F1 score (token-level): ", micro_f1)
            print("Macro F1 score (sentence-level): ", macro_f1)

        # Placeholder for DEP code
        elif self.task == 'dep':
            return 0

@dataclass
class CustomArguments:
    task: str = field(
        metadata={"help": "Specify the task name"}
    )
    model: str = field(
        metadata={"help": "Specify the model name from huggingface"}
    )
    tokenizer: str = field(
        metadata={"help": "Specify the tokenizer name from huggingface"}
    )
    data: str = field(
        metadata={"help": "Specify the data path (the folder that contains train.txt, dev.txt, and test.txt)"}
    )

def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True, help="Specify the task name")
    parser.add_argument("-m", "--model", type=str, required=True, help="Specify the model name from huggingface")
    parser.add_argument("-k", "--tokenizer", type=str, required=True, help="Specify the tokenizer name from huggingface")
    parser.add_argument("-d", "--data", type=str, required=True, help="Specify the data path (the folder that contains train.txt, dev.txt, and test.txt)")
    # Add TrainingArguments with custom defaults
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='Batch size per GPU/TPU core/CPU for training.')
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X updates steps.')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='Tensorboard log directory.')
    parser.add_argument('--logging_steps', type=int, default=500, help='Log every X updates steps.')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['no', 'steps', 'epoch'],
                        help='Evaluation strategy to use.')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate on dev set every X steps.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Evaluate on dev set every X steps.')
    """

    parser = HfArgumentParser((TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()

    #args = parser.parse_args()

    task = custom_args.task
    model_name = custom_args.model
    tokenizer_name = custom_args.tokenizer
    data_path = custom_args.data

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_eval = ModelEval(
        task=task,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        data_path=data_path,
        device=device,
        hf_args = training_args
    )

    model_eval.evaluate_model()

if __name__ == "__main__":
    main()
