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



def get_temporal_proposal_AR_AN(pr_list, score_list, gt_list, thresh=0.5, AN_max = 100):
    total_proposal_number = len(sum(pr_list,[]))
    total_video_number = len(gt_list)
    AN_submission = float(total_proposal_number)/float(total_video_number)
    R = float(AN_submission) / float(AN_max)
    proposals_modified_to_calculate = []
    if R < 1:
        for i in range(len(pr_list)):
            tmp = np.array(pr_list[i])
            sort_index = np.argsort(np.array(score_list[i]))
            sort_index = sort_index[::-1]
            cutoff = round(len(tmp) * (1-R))
            index_to_add = sort_index[0:int(cutoff)]
            tmp_to_add = tmp[index_to_add]
            proposals_modified_to_calculate.append(tmp_to_add)
    else:
        for i in range(len(pr_list)):
            tmp = np.array(pr_list[i])
            sort_index = np.argsort(np.array(score_list[i]))
            cutoff = round(len(tmp) * (R-1))
            index_to_add = sort_index[0:int(cutoff)]
            tmp_to_add = tmp + tmp[index_to_add]
            proposals_modified_to_calculate.append(tmp_to_add)
    
    pv_list = []
    pi_list = []
    for p in range(100):
        proposals_p = []
        for i in pr_list:
            indexend = 0.01 * (p+1) * len(i)
            proposals_p.append(i[0:int(round(indexend))])
        pv, pi = get_temporal_proposal_recall(proposals_p, gt_list, thresh)
        print('IoU threshold {:02f}, p {}.  per video recall: {:02f}, per instance recall: {:02f}'.format(thresh, p+1, pv * 100, pi * 100))
        pv_list.append(pv)
        pi_list.append(pi)
     
    return pv_list,pi_list
    






def cal_ap(rec, prec, use_11_point_metric=False):
    if use_11_point_metric:
        ap = 0.
        for t in np.arange(0, 1.01, 0.1):
            tmp = prec[rec >= t]
            if tmp.size == 0:
                p = 0.
            else:
                p = np.max(tmp)
            ap += p / 11
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = np.maximum(mpre[i], mpre[i + 1])
        i = np.where(mrec[1::] != mrec[:-1])[0] + 1
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap


def precision_recall_ap_eval(video_det, video_gt, cls, min_overlap=0.5, plot=False, debug=False,
                             use_11_point_metric=True):
    '''
        input: video_det: [[vid, pred, score, t1, t2], ..]
               video_gt:  [[vid, label, t1, t2], ..]
               min_overlap:
        output: mAP
    '''
    [det_vid, det_pred, det_score, det_seg1, det_seg2] = [list(x) for x in zip(*video_det)]
    [gt_vid, gt_label, gt_seg1, gt_seg2] = [list(x) for x in zip(*video_gt)]

    cls_idx = [i for i, x in enumerate(det_pred) if x == cls]
    det_vid = [det_vid[i] for i in cls_idx]
    det_score = [det_score[i] for i in cls_idx]
    det_seg1 = [det_seg1[i] for i in cls_idx]
    det_seg2 = [det_seg2[i] for i in cls_idx]

    # Sort detections by decreasing confidence
    order = np.array(det_score).argsort()[::-1]
    det_vid = [det_vid[i] for i in order]
    det_score = [det_score[i] for i in order]
    det_seg = [[det_seg1[i], det_seg2[i]] for i in order]

    # Turn ground-truth into expected format
    gt = {}
    for i in xrange(len(gt_vid)):
        if gt_label[i] != cls:
            continue
        if gt_vid[i] not in gt:
            gt[gt_vid[i]] = {}
            gt[gt_vid[i]]['segment'] = [[gt_seg1[i], gt_seg2[i]]]
            gt[gt_vid[i]]['detected'] = [-1]
            gt[gt_vid[i]]['iou'] = [-np.Inf]
        else:  # a video can have more than one annotated segment
            gt[gt_vid[i]]['segment'].append([gt_seg1[i], gt_seg2[i]])
            gt[gt_vid[i]]['detected'].append(-1)
            gt[gt_vid[i]]['iou'].append(-np.Inf)
    npos = 0
    for i in gt:
        npos += len(gt[i]['detected'])

        # Iterate over all detected regions, and compare with ground-truth
    nd = len(det_vid)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in xrange(nd):
        if not det_vid[d] in gt:
            fp[d] = 1
            continue
        vid = det_vid[d]
        t1 = det_seg[d][0]
        t2 = det_seg[d][1]
        IoU_max = -np.Inf

        for i in xrange(len(gt[vid]['segment'])):
            t1_gt = gt[vid]['segment'][i][0]
            t2_gt = gt[vid]['segment'][i][1]
            tt1 = np.maximum(t1, t1_gt)
            tt2 = np.minimum(t2, t2_gt)
            intersection = tt2 - tt1
            IoU = intersection / ((t2 - t1) + (t2_gt - t1_gt) - intersection)
            if IoU > IoU_max:
                IoU_max = IoU
                imax = i

        if IoU_max >= min_overlap:
            if gt[vid]['detected'][imax] == -1:
                tp[d] = 1.  # true positive
                gt[vid]['detected'][imax] = d
                gt[vid]['iou'][imax] = IoU_max
            else:
                fp[d] = 1.  # false positive(multiple detection)
        else:
            fp[d] = 1.  # false positive, also

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recall = tp_cum / npos
    precision = tp_cum / (tp_cum + fp_cum)

    ap = cal_ap(recall, precision, use_11_point_metric=use_11_point_metric)

    #
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('class: {}, AP = {:.3f}'.format(cls, ap))

    if debug:
        for vid in gt:
            print(vid)
            for d in xrange(nd):
                if det_vid[d] == vid:
                    print('{:5.1f}:{:5.1f} {:8.3f}'.format(det_seg[d][0], det_seg[d][1], det_score[d]))
            for i in range(len(gt[vid]['segment'])):
                p = gt[vid]['detected'][i]
                if p == -1:
                    print('groundtruth segment {:5.1f}:{:5.1f} missed'.format(gt[vid]['segment'][i][0],
                                                                              gt[vid]['segment'][i][1]))
                else:
                    print('groundtruth segment {:5.1f}:{:5.1f}, best detection {:5.1f}:{:5.1f}, with IOU={:3.3f}'.format(
                        gt[vid]['segment'][i][0], gt[vid]['segment'][i][1], det_seg[p][0], det_seg[p][1],
                        gt[vid]['iou'][i]))

    return recall, precision, ap
