from model_eval import ModelEval
import argparse
import torch
import os
import pandas as pd
from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class CustomArgumentsGeneral:
    hf_token: str = field(
        metadata={"help": "Specify your HuggingFace token to access a closed modek."}
    )
    model: str = field(
        metadata={"help": "Specify the model name from HuggingFace"}
    )
    tokenizer: str = field(
        metadata={"help": "Specify the tokenizer name from HuggingFace"}
    )
    data: str = field(
        metadata={"help": "Specify the data path that contains folders of all datasets in the same structure as 'scibert/data' fomr the SciBERT repository."}
    )
    max_length: int = field(
        metadata={"help": "Specify the maximum sequence length of the model to use when tokenizing."}
    )
    seq_to_seq_model: bool = field(
           default=False, 
           metadata={"help": ""})
    
    

def main():

    parser = HfArgumentParser((TrainingArguments, CustomArgumentsGeneral))
    training_args, custom_args = parser.parse_args_into_dataclasses()

    hf_token = custom_args.hf_token
    model_name = custom_args.model
    tokenizer_name = custom_args.tokenizer
    data_path = custom_args.data
    max_length = custom_args.max_length
    seq_to_seq_model = custom_args.seq_to_seq_model


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tasks_datasets_dict = {
    "ner": [
        data_path + "/ner/bc5cdr",
        data_path + "/ner/JNLPBA",
        data_path + "/ner/NCBI-disease", 
        data_path + "/ner/sciie"
        ],
    "pico": [
        data_path + "/pico/ebmnlp"
        ],
    "rel": [
        data_path + "/text_classification/chemprot",
        data_path + "/text_classification/sciie-relation-extraction"
        ],
    "cls": [
        data_path + "/text_classification/citation_intent",
        data_path + "/text_classification/mag",
        data_path + "/text_classification/sci-cite"
        ], 
    "dep": [
        data_path + "/parsing/genia"
    ]
    }

    tasks = []
    datasets = []
    metrics1 = []
    scores1 = []
    metrics2 = []
    scores2 = []

    for key, value in tasks_datasets_dict.items():

        print("Evaluating " + key + "...")

        for data_path_value in value:

            print("Using " + data_path_value + "...")
            
            model_eval = ModelEval(
                task=key,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                data_path=data_path_value,
                device=device,
                hf_args = training_args, 
                hf_token = hf_token,
                max_length = max_length,
                s2s_model = seq_to_seq_model
            )
            
            score1, score2 = model_eval.evaluate_model()

            directories = data_path_value.split(os.path.sep)
            dataset_name = directories[-1]

            if key == 'dep':
                metric1 = 'uas'
                metric2 = 'las'
            else:
                metric1 = 'micro F1'
                metric2 = 'macro F1'

            tasks.append(key)
            datasets.append(dataset_name)
            metrics1.append(metric1)
            scores1.append(score1)
            metrics2.append(metric2)
            scores2.append(score2)

    df_dict = {
        'task' : tasks,
        'dataset': datasets,
        'metric 1': metrics1,
        'score 1': scores1,
        'metric 2': metrics2, 
        'score 2': scores2
    }

    results_df = pd.DataFrame(df_dict)

    current_dir = os.getcwd()
    model = model_name.split('/')[-1]

    results_df.to_csv(current_dir + "/" + model + ".csv")


if __name__ == "__main__":
    main()
