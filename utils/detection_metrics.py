"""
This module provides some utils for calculating metrics in temporal action detection
"""
import numpy as np


def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])


def overlap_over_b(span_A, span_B):
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])
    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(span_B[1] - span_B[0])


def temporal_recall(gt_spans, est_spans, thresh=0.5):
    """
    Calculate temporal recall of boxes and estimated boxes
    Parameters
    ----------
    gt_spans: [(start, end), ...]
    est_spans: [(start, end), ...]

    Returns
    recall_info: (hit, total)
    -------

    """
    hit_slot = [False] * len(gt_spans)
    for i, gs in enumerate(gt_spans):
        for es in est_spans:
            if temporal_iou(gs, es) > thresh:
                hit_slot[i] = True
                break
    recall_info = (np.sum(hit_slot), len(hit_slot))
    return recall_info


def name_proposal(gt_spans, est_spans, thresh=0.0):
    """
    Assigng label to positive proposals
    :param gt_spans: [(label, (start, end)), ...]
    :param est_spans: [(start, end), ...]
    :param thresh:
    :return: [(label, overlap, start, end), ...] same number of est_spans
    """
    ret = []
    for es in est_spans:
        max_overlap = 0
        max_overlap_over_self = 0
        label = 0
        for gs in gt_spans:
            ov = temporal_iou(gs[1], es)
            ov_pr = overlap_over_b(gs[1], es)
            if ov > thresh and ov > max_overlap:
                label = gs[0] + 1
                max_overlap = ov
                max_overlap_over_self = ov_pr
        ret.append((label, max_overlap, max_overlap_over_self, es[0], es[1]))

    return ret


def get_temporal_proposal_recall(pr_list, gt_list, thresh):
    recall_info_list = [temporal_recall(x, y, thresh=thresh) for x, y in zip(gt_list, pr_list)]
    per_video_recall = np.sum([x[0] == x[1] for x in recall_info_list]) / float(len(recall_info_list))
    per_inst_recall = np.sum([x[0] for x in recall_info_list]) / float(np.sum([x[1] for x in recall_info_list]))
    return per_video_recall, per_inst_recall

