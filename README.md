## Overview

A framework for evaluating HuggingFace models on specific tasks by fine-tuning on given datasets.

## Running script

In order to get the evaluation result, run the ```model_eval.py``` script with the following arguments:


* ```-t``` or``` --task```: the task at hand, chosen from a specific list (see below).
  
* ```-m``` or ```--model```: the model name as written on HuggingFace.

* ```-k``` or ```--tokenizer```: the tokenizer name as written on HuggingFace.

* ```-d``` or ```--data```: the path to the data folder that contains train.txt, dev.txt, and test.txt.


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
| Bio   | NER  | BC5CDR       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.97407 |
|       |      | JNLPBA       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.95741 |
|       |      | NCBI-disease | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.92645 |
|       | PICO | EBMNLP       | allenai/scibert_scivocab_uncased | Macro F1 (token-level) | 0.78952 |
|       | DEP  | GENIA-LAS    |                                  |                        |         |
|       |      | GENIA-UAS    |                                  |                        |         |
|       | REL  | ChemProt     | allenai/scibert_scivocab_uncased | Micro F1 (sentence-level)| 0.56720|
| CS    | NER  | SciERC       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.756007 |
|       | REL  | SciERC       | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.80679|
|       | CLS  | ACL-ARC      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.71327|
| Multi | CLS  | Paper Field  | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.73595|
|       | CLS  | SciCite      | allenai/scibert_scivocab_uncased | Macro F1 (sentence-level)| 0.85118|
