import torch
from datasets import DatasetDict
from typing import Tuple, List


def calculate_uas_las(
    tokens: List,
    gold_heads: List,
    gold_dep_labels: List,
    pred_heads: List,
    pred_dep_labels: List,
) -> Tuple[float, float]:
    """Calculates the labeled attachment score (LAS) and the unlabeled attachment score (UAS).
    Args:
        tokens (List): List of tokens from the test set.
        gold_heads (List): List of gold head indices.
        gold_dep_labels (List): List of gold dependency labels.
        pred_heads (List): List of predicted head indices.
        pred_dep_labels (List): List of predicted dep labels.
    Returns:
        Tuple[float, float]: LAS and UAS scores.
    """
    total_tokens = len(tokens)
    uas = 0
    las = 0

    for g_heads, g_labels, p_heads, p_labels in zip(
        gold_heads, gold_dep_labels, pred_heads, pred_dep_labels
    ):
        for g_head, g_label, p_head, p_label in zip(
            g_heads, g_labels, p_heads, p_labels
        ):
            if g_head == p_head:
                uas += 1
                if g_label == p_label:
                    las += 1

    uas_score = uas / total_tokens
    las_score = las / total_tokens

    return uas_score, las_score


def collect_predictions_dep(
    dataset_dict: DatasetDict, model, label_map: set, device
) -> List:
    predictions = []

    for sample in dataset_dict:
        inputs = {
            key: torch.tensor(sample[key]).unsqueeze(0).to(device)
            for key in sample.keys()
            if key != "labels"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            for logit in logits:
                pred_label_ids = torch.argmax(logit, dim=-1).tolist()
                pred_label = [label_map[label_id] for label_id in pred_label_ids]
                predictions.append(pred_label)
    return predictions


def collect_gold_labels_dep(dataset_dict: DatasetDict, label_map: set) -> List:
    gold_labels = []

    for sample in dataset_dict:
        labels = [label_map[label_id] for label_id in sample]
        gold_labels.append(labels)

    return gold_labels
