## Overview

A framework for evaluating HuggingFace models on specific tasks by fine-tuning on given datasets.

## Running script

In order to get the evaluation result, run the ```model_eval.py``` script with the following required arguments:


* ``` --task```: the task at hand, chosen from a specific list (see below).
  
* ```--model```: the model name as written on HuggingFace.

* ```--tokenizer```: the tokenizer name as written on HuggingFace.

* ```--data```: the path to the data folder that contains train.txt, dev.txt, and test.txt.

* ```--output_dir```: the directory to save the model and its results.

In addition, all arguments for [HuggingFace's TrainingArgument](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) class can be given, otherwise default values will be used.

## Available tasks

The following tasks are available for evaluation:

* ```ner```: Named Entity Recognition. NER tasks are evaluated using Macro F1 on the span level.

* ```pico```:  A sequence labeling task where the model extracts spans describing the Participants, Interventions, Comparisons, and Outcomes (PICO) in a clinical trial paper.
Evaluated using macro-F1 on the token level.

* ```dep```:

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
| Bio   | NER  | BC5CDR       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.94378 |
|       |      | JNLPBA       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.93917 |
|       |      | NCBI-disease | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.88986 |
|       | PICO | EBMNLP       | allenai/scibert_scivocab_uncased | Macro F1 (token-level) | 0.78838 |
|       |      |              |                                   | Micro F1 (token-level) | 0.97080 |
|       | DEP  | GENIA-LAS    |                                  |                        |         |
|       |      | GENIA-UAS    |                                  |                        |         |
|       | REL  | ChemProt     | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.57377|
|       |       |             |                                  | Micro F1 (sentence-level)| 0.84607|
| CS    | NER  | SciERC       | allenai/scibert_scivocab_cased | Macro F1 (span-level)  | 0.62328 |
|       | REL  | SciERC       | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.78592|
|       |       |              |                                 | Micro F1 (sentence-level)| 0.86242|
|       | CLS  | ACL-ARC      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.59925|
|       |       |             |                                  | Micro F1 (sentence-level)| 0.76978|
| Multi | CLS  | Paper Field  | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.74535|
|        |     |              |                                  | Micro F1 (sentence-level)| 0.74602|
|       | CLS  | SciCite      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.85473|
|       |      |              |                                  | Micro F1 (sentence-level)| 0.86674|


All results above were trained using the following hyperparameters: 
* Number of epochs: 4
* Batch size: 32
* Learning rate: 2e-5

All other hyperparameters are the default TrainingArguments values. 

NER models were trained using the cased SciBERT variant to match the results resulted in the SciBERT paper, all other tasks were trained using the uncases model variant. 
