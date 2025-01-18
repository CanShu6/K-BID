import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res.mean()


def mrr_at_k_batch(hits, k):
    """
    calculate MRR@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k] / np.arange(1, k + 1)).sum(axis=1)
    return res.mean()


def auc_at_k(ground_truth, prediction):
    """
    calculate AUC@k
    ground_truth: array, element is binary (0 / 1), 1-dim
    prediction: array, element is the predicted score, 1-dim
    """
    try:
        res = roc_auc_score(y_true=ground_truth[1:], y_score=prediction[ground_truth[0]])
    except ValueError:
        res = 0.
    return res


def auc_at_k_batch(ground_truth, prediction, k):
    """
    calculate AUC@k
    ground_truth: array, element is binary (0 / 1), 2-dim
    prediction: array, element is the predicted score, 2-dim
    """
    ground_truth = ground_truth[:, :k].astype(np.int8)
    prediction = prediction[:, :k]
    idx = np.arange(0, ground_truth.shape[0]).reshape(-1, 1)
    ground_truth = np.hstack([idx, ground_truth])
    res = np.apply_along_axis(auc_at_k, 1, ground_truth, prediction=prediction)
    return res.mean()


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    sorted_scores, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['mrr'] = mrr_at_k_batch(binary_hit, k)
        metrics_dict[k]['auc'] = auc_at_k_batch(binary_hit, sorted_scores.cpu().numpy(), k)
    return metrics_dict


