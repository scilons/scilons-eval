from data.data_prep_utils import tokenize_function
from transformers import AutoTokenizer
import torch


def test_prepare_input_rel_cls():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    texts = [
        "This paper presents an [[ algorithm ]] for << computing optical flow , shape , motion , lighting",
        "The algorithm utilizes both spatial and temporal intensity variation as << cues >>",
    ]

    tokenized_texts = tokenize_function(texts, tokenizer)

    labels = ["USED-FOR", "FEATURE-OF"]
    labels_set = set(labels)

    labels_mapper = {label: i for i, label in enumerate(labels_set)}

    token_ids = tokenized_texts["input_ids"].to(device)
    attention_masks = tokenized_texts["attention_mask"].to(device)
    token_type_ids = tokenized_texts["token_type_ids"].to(device)
    labels_encoded = [labels_mapper[label] for label in labels]

    assert torch.is_tensor(token_ids)
    assert torch.is_tensor(attention_masks)
    assert torch.is_tensor(token_type_ids)
    assert isinstance(labels_encoded, list)
    assert labels_encoded == [0, 1]
