import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(label, pred, k):
    if label in pred[:k]:
        return 1. / k
    else:
        return 0.


def mrr_at_k(label, pred, k):
    try:
        return 1. / (pred[:k].index(label) + 1)
    except ValueError:
        return 0.


def auc_at_k(truth, scores, k):
    try:
        return roc_auc_score(y_true=truth[:k], y_score=scores[:k])
    except ValueError:
        return 0.


def calc_metrics_at_k(label, voter_scores: list, voter_ids: list, Ks):
    """
    计算 Precision, MRR 和 AUC
    :param label: 真实回流者
    :param voter_scores: 模型对回流者的打分
    :param voter_ids: 参与模型打分的回流者
    :param Ks: top k
    """
    metrics_dict = {}
    try:
        label_idx = voter_ids.index(label)
        sorted_idx = np.argsort(voter_scores)[::-1].tolist()
        sorted_scores = np.sort(voter_scores)[::-1].tolist()

        ground_truth = [0] * len(voter_ids)
        ground_truth[sorted_idx.index(label_idx)] = 1
        for k in Ks:
            metrics_dict[k] = {}
            metrics_dict[k]['precision'] = precision_at_k(label_idx, sorted_idx, k)
            metrics_dict[k]['mrr'] = mrr_at_k(label_idx, sorted_idx, k)
            metrics_dict[k]['auc'] = auc_at_k(ground_truth, sorted_scores, k)
    except ValueError:
        for k in Ks:
            metrics_dict[k] = {}
            metrics_dict[k]['precision'] = 0.
            metrics_dict[k]['mrr'] = 0.
            metrics_dict[k]['auc'] = 0.
    return metrics_dict
