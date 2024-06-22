## Overview

A framework for evaluating HuggingFace models on specific tasks by fine-tuning on given datasets.

## Running script

#### To get the evaluation result for one specific task and dataset, run the ```model_eval.py``` script with the following required arguments:


* ``` --task```: the task at hand, chosen from a specific list (see below).
  
* ```--model```: the model name as written on HuggingFace.

* ```--tokenizer```: the tokenizer name as written on HuggingFace.

* ```--data```: the path to the data folder that contains train.txt, dev.txt, and test.txt.

* ```--hf_token```: your access token from HuggingFace.

* ```--max_length```: max sequence length of the model for tokenization.

* ```--output_dir```: the directory to save the model and its results.

#### To get the evaluation result for all tasks on all available dataset (see below), run the ```evaluate_all_tasks.py``` script. The code will save a .csv file with the results in the directory from which you ran the script. The following arguments are required:

* ```--model```: the model name as written on HuggingFace.

* ```--tokenizer```: the tokenizer name as written on HuggingFace.

* ```--data```: the path to the folder that contains all required datasets (such as https://github.com/allenai/scibert/tree/master/data).

* ```--hf_token```: your access token from HuggingFace.

* ```--max_length```: max sequence length of the model for tokenization.

* ```--output_dir```: the directory to save the model and its results.


In addition, all arguments for [HuggingFace's TrainingArgument](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) class can be given, otherwise default values will be used.


## Available tasks

The following tasks are available for evaluation:

* ```ner```: Named Entity Recognition. NER tasks are evaluated using Macro F1 on the span level.

* ```pico```:  A sequence labeling task where the model extracts spans describing the Participants, Interventions, Comparisons, and Outcomes (PICO) in a clinical trial paper.
Evaluated using macro-F1 on the token level.

* ```dep```: A task that analyzes the grammatical structure of a sentence to identify relationships between words. Evaluated using Unlabeled Attachment Scores (UAS), which measures the percentage of correctly identified head-dependent pairs, and Labeled Attachment Score (LAS), which accounts for the correct labeling of these dependencies. 

* ```rel```: Relation Classication. The model predicts the type of relation expressed between two entities, which are encapsulated in the sentence by inserted special tokens. REL tasks are evaluated using Macro F1 on the sentence level.

* ```cls```: Text Classification. CLS tasks are evaluated using Macro F1 on the sentence level.

## Available datasets

* BC5CDR (Li et al., 2016) for NER
* JNLPBA (Collier and Kim, 2004) for NER
* NCBI-disease (Dogan et al., 2014) for NER
* EBM-NLP (Nye et al., 2018) for PICO
* GENIA (Kim et al., 2003) - LAS for DEP
* GENIA (Kim et al., 2003) - UAS for DEP
* ChemProt (Kringelum et al., 2016) for REL
* SciERC (Luan et al., 2018) for REL and NER
* ACL-ARC (Jurgens et al., 2018) for CLS
* Paper Field Classification (from the Microsoft Academic Graph data) for CLS
* SciCite (Cohan et al., 2019) for CLS

All datasets are available on the [SciBERT GitHub repository](https://github.com/allenai/scibert/tree/master/data) and can be directly used from there.

## Results on SciBERT

| Field | Task | Dataset      | Model Name                       | Metric                 | Result  |
|-------|------|--------------|----------------------------------|------------------------|---------|
| Bio   | NER  | BC5CDR       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.94052 |
|       |      | JNLPBA       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.93575 |
|       |      | NCBI-disease | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.87896 |
|       | PICO | EBMNLP       | allenai/scibert_scivocab_uncased | Macro F1 (token-level) | 0.79029 |
|       |      |              |                                   | Micro F1 (token-level) | 0.97099 |
|       | DEP  | GENIA-LAS    | allenai/scibert_scivocab_cased  | LAS | 0.3875|
|       |      | GENIA-UAS    |                                   | UAS | 0.44729|
|       | REL  | ChemProt     | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.57724|
|       |       |             |                                  | Micro F1 (sentence-level)| 0.84837|
| CS    | NER  | SciERC       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.51104 |
|       | REL  | SciERC       | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.79376|
|       |       |              |                                 | Micro F1 (sentence-level)| 0.87166|
|       | CLS  | ACL-ARC      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.61125|
|       |       |             |                                  | Micro F1 (sentence-level)| 0.76978|
| Multi | CLS  | Paper Field  | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.74419|
|        |     |              |                                  | Micro F1 (sentence-level)| 0.74522|
|       | CLS  | SciCite      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.85621|
|       |      |              |                                  | Micro F1 (sentence-level)| 0.86781|


All results above were trained using the following hyperparameters: 
* Number of epochs: 4
* Batch size: 32
* Learning rate: 2e-5

All other hyperparameters are the default TrainingArguments values. 

NER and DEP models were trained using the cased SciBERT variant to match the results in the SciBERT paper. All other tasks were trained using the uncased model variant. DEP results are calculated without removing punctuation. 
