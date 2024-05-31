from model_eval import ModelEval
import argparse
from transformers import HfArgumentParser

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
    

def main():

    parser = HfArgumentParser((TrainingArguments, CustomArgumentsGeneral))
    training_args, custom_args = parser.parse_args_into_dataclasses()

    hf_token = custom_args.hf_token
    model_name = custom_args.model
    tokenizer_name = custom_args.tokenizer
    data_path = custom_args.data

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
        ]
    }

    for key, value in tasks_datasets_dict.items():

        print("Evaluating " + key + "...")

        for data_path_value in value:

            print("Using" + data_path_value + "...")
            
            model_eval = ModelEval(
                task=key,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                data_path=data_path_value,
                device=device,
                hf_args = training_args, 
                hf_token = hf_token
            )
            
            model_eval.evaluate_model()


if __name__ == "__main__":
    main()