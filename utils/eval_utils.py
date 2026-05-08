import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def hungary_iou(dists, dist_thresh=0.1):
    distances = dists.copy()
    gt_l = distances.shape[0]
    pred_l = distances.shape[1]
    indice = linear_sum_assignment(distances)
    distances = distances[indice[0], indice[1]]
    tp = np.sum(distances <= dist_thresh)
    return tp / (gt_l + pred_l - tp)


def get_cd(gts, pts):
    dists = cdist(gts, pts, metric='euclidean')
    cd = np.sum(np.min(dists, axis=1)) / dists.shape[0] + np.sum(np.min(dists.T, axis=1)) / dists.shape[1]
    return cd

def get_cd_dist(dists):
    cd = np.sum(np.min(dists, axis=1)) / dists.shape[0] + np.sum(np.min(dists.T, axis=1)) / dists.shape[1]
    return cd

def get_iou(dists, dist_thresh=0.1):
    """
    dists: (N_gt, N_pred) pairwise distance matrix
    return: IoU score
    """

    gt_l, pred_l = dists.shape

    gt_match = np.min(dists, axis=1) <= dist_thresh
    pred_match = np.min(dists, axis=0) <= dist_thresh

    tp = np.sum(gt_match)
    iou = tp / (gt_l + pred_l - tp + 1e-8)

    return float(iou)

def get_geo_iou(dists, dist_thresh):
    """
    Computes (npos, fp, fn) from a distance matrix using the *existing* testt.py convention.

    dists: (N_rows, N_pred) matrix
      - In the original code this is built as: dists = geo_dists[:, pred_indices]
      - So N_rows is whatever geo_dists rows represent (commonly N_gt keypoints in this repo).
    """
    if dists.ndim != 2:
        raise ValueError(f"dists must be 2D, got shape={dists.shape}")
    npos = int(dists.shape[0])
    if npos == 0:
        return 0, 0, 0
    if dists.shape[1] == 0:
        return npos, 0, npos
    fp = int(np.count_nonzero(np.all(dists > dist_thresh, axis=0)))
    fn = int(np.count_nonzero(np.all(dists > dist_thresh, axis=1)))

    denom = max(npos + fp, np.finfo(np.float64).eps)
    iou = float((npos - fn) / denom)

    return iou

def get_pck(gt_cls, pred_cls, geo_dists, dist_thresh):
    correct = 0
    G = geo_dists.shape[0]

    for i in range(G):
        valid = np.where(pred_cls == gt_cls[i])[0]
        if len(valid) == 0:
            continue
        min_dist = geo_dists[i, valid].min()
        if min_dist < dist_thresh:
            correct += 1

    return correct / G

def get_DAS(dists, gt_labels, pred_labels):
    gt_close_idx = np.argmin(dists, axis=1)
    gt_pred_labels = pred_labels[gt_close_idx]
    acc_1 = np.mean(gt_labels == gt_pred_labels)

    pred_clos_idx = np.argmin(dists, axis=0)
    pred_gt_idx = gt_labels[pred_clos_idx]
    acc_2 = np.mean(pred_labels == pred_gt_idx)

    acc = (acc_1 + acc_2)/2

    return acc