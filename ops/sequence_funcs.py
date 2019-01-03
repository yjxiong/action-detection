from .metrics import softmax

import sys
import numpy as np
from scipy.ndimage import gaussian_filter
try:
    from nms.nms_wrapper import nms
except ImportError:
    nms = None

def label_frame_by_threshold(score_mat, cls_lst, bw=None, thresh=list([0.05]), multicrop=True):
    """
    Build frame labels by thresholding the foreground class responses
    :param score_mat:
    :param cls_lst:
    :param bw:
    :param thresh:
    :param multicrop:
    :return:
    """
    if multicrop:
        f_score = score_mat.mean(axis=1)
    else:
        f_score = score_mat

    ss = softmax(f_score)

    rst = []
    for cls in cls_lst:
        cls_score = ss[:, cls+1] if bw is None else gaussian_filter(ss[:, cls+1], bw)
        for th in thresh:
            rst.append((cls, cls_score > th, f_score[:, cls+1]))

    return rst


def gen_exponential_sw_proposal(video_info, time_step=1, max_level=8, overlap=0.4):
    spans = [2 ** x for x in range(max_level)]
    duration = video_info.duration
    pr = []
    for t_span in spans:
        span = t_span * time_step
        step = int(np.ceil(span * (1 - overlap)))
        local_boxes = [(i, i + t_span) for i in np.arange(0, duration, step)]
        pr.extend(local_boxes)

    # fileter proposals
    # a valid proposal should have at least one second in the video
    def valid_proposal(duration, span):
        real_span = min(duration, span[1]) - span[0]
        return real_span >= 1

    pr = list(filter(lambda x: valid_proposal(duration, x), pr))
    return pr


def temporal_nms(bboxes, thresh, score_ind=3):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, cls, score], ...]
    :param thresh:
    :return:
    """
    if not nms:
        return temporal_nms_fallback(bboxes, thresh, score_ind=score_ind)
    else:
        keep = nms(np.array([[x[0], x[1], x[3]] for x in bboxes]), thresh, device_id=0)
        return [bboxes[i] for i in keep]


def temporal_nms_fallback(bboxes, thresh, score_ind=3):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, cls, score], ...]
    :param thresh:
    :return:
    """
    t1 = np.array([x[0] for x in bboxes])
    t2 = np.array([x[1] for x in bboxes])
    scores = np.array([x[score_ind] for x in bboxes])

    durations = t2 - t1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1 + 1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return [bboxes[i] for i in keep]



def build_box_by_search(frm_label_lst, tol, min=1):
    boxes = []
    for cls, frm_labels, frm_scores in frm_label_lst:
        length = len(frm_labels)
        diff = np.empty(length+1)
        diff[1:-1] = frm_labels[1:].astype(int) - frm_labels[:-1].astype(int)
        diff[0] = float(frm_labels[0])
        diff[length] = 0 - float(frm_labels[-1])
        cs = np.cumsum(1 - frm_labels)
        offset = np.arange(0, length, 1)

        up = np.nonzero(diff == 1)[0]
        down = np.nonzero(diff == -1)[0]

        assert len(up) == len(down), "{} != {}".format(len(up), len(down))
        for i, t in enumerate(tol):
            signal = cs - t * offset
            for x in range(len(up)):
                s = signal[up[x]]
                for y in range(x + 1, len(up)):
                    if y < len(down) and signal[up[y]] > s:
                        boxes.append((up[x], down[y-1]+1, cls, sum(frm_scores[up[x]:down[y-1]+1])))
                        break
                else:
                    boxes.append((up[x], down[-1] + 1, cls, sum(frm_scores[up[x]:down[-1] + 1])))

            for x in range(len(down) - 1, -1, -1):
                s = signal[down[x]] if down[x] < length else signal[-1] - t
                for y in range(x - 1, -1, -1):
                    if y >= 0 and signal[down[y]] < s:
                        boxes.append((up[y+1], down[x] + 1, cls, sum(frm_scores[up[y+1]:down[x] + 1])))
                        break
                else:
                    boxes.append((up[0], down[x] + 1, cls, sum(frm_scores[0:down[x]+1 + 1])))

    return boxes
