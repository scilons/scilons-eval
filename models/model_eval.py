from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from data.prepare_data import DatasetPrep
from data.data_prep_utils import extract_spans


class ModelEval:

    def __init__(self, task, model_name, tokenizer_name, data_path, device):
        self.task = task
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.device = device


    def train_model(self):

        if self.task == 'ner':

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
            
        else:
            return None

    
    def evaluate_model(self):

        trained_model, dataset_dic, labels_mappert = self.train_model()
        trained_model.eval()

        all_predictions = []
        all_labels = []

        for sample in dataset_dict['test']:
            inputs = {key: torch.tensor(sample[key]).unsqueeze(0).to(self.device) for key in sample.keys() if key != 'labels'}
            labels = torch.tensor(sample['labels']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = trained_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                
            true_spans = extract_spans(labels)
            predicted_spans = extract_spans(predictions)

            for span in true_spans:
                true_spans_by_type[span["type"]].append(span)
    
            for span in predicted_spans:
                predicted_spans_by_type[span["type"]].append(span)


        return all_predictions, all_labels

        # Flatten the lists of predictions and true labels
        flat_predictions = np.concatenate(all_predictions, axis=0)
        flat_labels = np.concatenate(all_labels, axis=0)

        # Compute macro F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(flat_labels, flat_predictions, average='macro')

        return precision, recall, f1