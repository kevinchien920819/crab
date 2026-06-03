import numpy as np

# ref: https://github.com/TakHemlata/SSL_Anti-spoofing/blob/4acaa61dcef5f7610f43aa4d0b29c4559b970cd2/eval_metric_LA.py#L21

def compute_det_curve(target_scores, nontarget_scores):
    """Compute false rejection, false acceptance, and threshold arrays for DET analysis."""
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_eer_from_labels(true, pred):
    """Compute EER after splitting scores by binary ground-truth labels."""
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)

    if true.size == 0 or pred.size == 0:
        raise ValueError('empty labels or predictions')
    if true.shape != pred.shape:
        raise ValueError('labels and predictions must have the same shape')
    if np.unique(true).size < 2:
        raise ValueError('labels contain only one class')

    target_scores = pred[true == 1]
    nontarget_scores = pred[true == 0]
    eer, _ = compute_eer(target_scores, nontarget_scores)
    return eer


# new feature for emotion recognition evaluation
def compute_uar(true, pred):
    """Compute unweighted average recall over the classes present in the labels."""
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    classes = np.unique(true)
    recalls = []
    for c in classes:
        true_mask = (true == c)
        recall = np.sum((pred == c) & true_mask) / np.sum(true_mask)
        recalls.append(recall)
    return np.mean(recalls)

def compute_war(true, pred):
    """Compute weighted average recall as overall accuracy."""
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    return np.sum(true == pred) / len(true)
