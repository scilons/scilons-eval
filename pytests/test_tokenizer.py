from data.data_prep_utils import tokenize_function
from transformers import AutoTokenizer
from transformers import BatchEncoding
import torch


def test_tokenizer_function():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    texts = [
        "This paper presents an [[ algorithm ]] for << computing optical flow , shape , motion , lighting",
        "The algorithm utilizes both spatial and temporal intensity variation as << cues >>",
    ]

    tokenized_texts = tokenize_function(texts, tokenizer)

    assert isinstance(tokenized_texts, BatchEncoding)
    assert "input_ids" in tokenized_texts
    assert torch.is_tensor(tokenized_texts["input_ids"])
    assert "token_type_ids" in tokenized_texts
    assert torch.is_tensor(tokenized_texts["token_type_ids"])
    assert "attention_mask" in tokenized_texts
    assert torch.is_tensor(tokenized_texts["attention_mask"])
