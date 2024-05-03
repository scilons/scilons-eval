## Overview

A framework for evaluating HuggingFace models on specific tasks by fine-tuning on given datasets.

## Running script

The following arguments should be given to run the script:


* ```-t``` or``` --task```: the task at hand, chosen from a specific list (see below).
  
* ```-m``` or ```--model```: the model name as written on HuggingFace.

* ```-k``` or ```--tokenizer```: the tokenizer name as written on HuggingFace.

* ```-d``` or ```--data```: the path to the data folder that contains train.txt, dev.txt, and test.txt.


## Available tasks

The following tasks are available for evaluation:

* ```ner```: Named Entity Recognition. NER tasks are evaluated using Macro F1 on the span level.

* ```pico```:  A equence labeling task where the model extracts spans describing the Participants, Interventions, Comparisons, and Outcomes (PICO) in a clinical trial paper.
Evaluated using macro-F1 on the token evel.

* ```dep```:

* ```rel```:

* ```cls```:

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
* SciCite (Cohan et al., 2019) for CLS


## Results on SciBERT

| Field | Task | Dataset      | Model Name                       | Metric                 | Result             |
|-------|------|--------------|----------------------------------|------------------------|--------------------|
| Bio   | NER  | BC5CDR       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.9947334962764466 |
|       |      | JNLPBA       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.9718915854922392 |
|       |      | NCBI-disease | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.9823094237515403 |
|       | PICO | EBMNLP       | allenai/scibert_scivocab_uncased | Macro F1 (token-level) | 0.7925765849220929 |
|       | DEP  | GENIA-LAS    |                                  |                        |                    |
|       |      | GENIA-UAS    |                                  |                        |                    |
|       | REL  | ChemProt     |                                  |                        |                    |
| CS    | NER  | SciERC       | allenai/scibert_scivocab_uncased | Macro F1 (span-level)  | 0.8541137319732242 |
|       | REL  | SciERC       |                                  |                        |                    |
|       | CLS  |              |                                  |                        |                    |
