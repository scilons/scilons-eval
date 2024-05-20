from data.data_prep_utils import tokenize_function, prepare_input_rel_cls
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

    token_ids, attention_masks, token_type_ids, labels = prepare_input_rel_cls(
        tokenized_texts, labels_mapper, labels, device
    )

    assert torch.is_tensor(token_ids)
    assert torch.is_tensor(attention_masks)
    assert torch.is_tensor(token_type_ids)
    assert isinstance(labels, list)
    assert labels == [0, 1]
