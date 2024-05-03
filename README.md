## Overview

A framework for evaluating HuggingFace models on specific tasks by fine-tuning on given datasets.

## Running script

The following arguments should be given to run the script:

```
-t or --task: the task at hand, chosen from a specific list of ‘ner’, ‘pico’, ‘rel’, ‘dep’, ..
-m or --model: the model name as written on huggingface
-k or --tokenizer: the tokenizer name as written on huggingface
-d or --data: the path to the data folder that contains train.txt, dev.txt, and test.txt 
```

## Available tasks

```
'ner', 'pico', 'dep', 'rel', and 'cls'.
```

## Available datasets



## Results on SciBERT
