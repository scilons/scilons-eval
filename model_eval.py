from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from data.prepare_data import DatasetPrep
from data.data_prep_utils import extract_spans
from collections import defaultdict
import argparse
import torch 

class ModelEval:

    def __init__(self, task, model_name, tokenizer_name, data_path, device):
        self.task = task
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.device = device


    def train_model(self):

        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        dataset_prep = DatasetPrep(
            task=self.task, 
            data_path=self.data_path, 
            tokenizer=tokenizer, 
            device=self.device
            )   
        dataset_dict, labels_mapper = dataset_prep.run()
            
        model = BertForTokenClassification.from_pretrained(self.model_name, num_labels=len(labels_mapper))
        model = model.to(self.device)
            
           
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            report_to="wandb",
            logging_steps=100,
            save_steps=500, 
            evaluation_strategy="steps",
            eval_steps=500,
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'], 
            eval_dataset=dataset_dict['dev']
            )

        trainer.train()

        trainer.save_model("results/models")

        return model, dataset_dict, labels_mapper
            
    
    def evaluate_model(self):

        trained_model, dataset_dict, labels_mapper = self.train_model()
        trained_model.eval()

        # NER is evaluated according to span-level macro F1
        if self.task == 'ner':

            true_spans_by_type = defaultdict(list)
            predicted_spans_by_type = defaultdict(list)
            
            for sample in dataset_dict['test']:
                inputs = {key: torch.tensor(sample[key]).unsqueeze(0).to(self.device) for key in sample.keys() if key != 'labels'}
                labels = sample["labels"]
            
                with torch.no_grad():
                    outputs = trained_model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()
                    
                true_spans = extract_spans(labels, labels_mapper)
                predicted_spans = extract_spans(predictions, labels_mapper)
    
                for span in true_spans:
                    true_spans_by_type[span["type"]].append(span)
            
                for span in predicted_spans:
                    predicted_spans_by_type[span["type"]].append(span)
                
        
            # Compute F1 score for each span type
            span_types = set(list(true_spans_by_type.keys()) + list(predicted_spans_by_type.keys()))
            macro_f1_scores = []
    
            for span_type in span_types:
                true_positives = sum(1 for span in predicted_spans_by_type[span_type] if span in true_spans_by_type[span_type])
                false_positives = sum(1 for span in predicted_spans_by_type[span_type] if span not in true_spans_by_type[span_type])
                false_negatives = sum(1 for span in true_spans_by_type[span_type] if span not in predicted_spans_by_type[span_type])
            
                precision = true_positives / (true_positives + false_positives + 1e-9)
                recall = true_positives / (true_positives + false_negatives + 1e-9)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
                
                macro_f1_scores.append(f1)
    
            macro_f1_score = sum(macro_f1_scores) / len(macro_f1_scores)
            
            return macro_f1_score

        # PICO is evaluated based on token-level macro-F1
        elif self.task == 'pico':

            predictions = []
            true_labels = []

            for sample in dataset_dict['test']:
                
                inputs = {key: torch.tensor(sample[key]).unsqueeze(0).to(self.device) for key in sample.keys() if key != 'labels'}
                labels = sample["labels"]
                
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_labels = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()

            # Extend lists
            predictions.extend(predicted_labels)
            true_labels.extend(labels)

            # Compute metrics
            macro_f1 = f1_score(true_labels, predictions, average='macro')
            macro_precision = precision_score(true_labels, predictions, average='macro')
            macro_recall = recall_score(true_labels, predictions, average='macro')

            return macro_f1
        
        # REL and CLS are evaluated based on sentence-level macro-F1
        elif self.task == 'cls' or self.task == 'rel':
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, help='Specify the task name')
    parser.add_argument('-m', '--model', type=str, help='Specify the model name from huggingface')
    parser.add_argument('-k', '--tokenizer', type=str, help='Specify the tokenizer name from huggingface')
    parser.add_argument('-d', '--data', type=str, help='Specify the data path (the folder that contains train.txt, dev.txt, and test.txt)')

    args = parser.parse_args()

    task = args.task
    model_name = args.model
    tokenizer_name = args.tokenizer
    data_path = args.data

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_eval = ModelEval(
        task=task, 
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        data_path=data_path,
        device=device
        )

    eval_score = model_eval.evaluate_model()

    if task == 'ner':
        print("Macro F1 score (span-level): ", eval_score)
    elif task == 'pico':
        print("Macro F1 score (token-level): ", eval_score)

    
if __name__ == "__main__":
    main()