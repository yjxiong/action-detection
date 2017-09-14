import argparse
import time

import numpy as np

from ssn_dataset import SSNDataSet
from transforms import *
from ops.utils import temporal_nms
import pandas as pd
from multiprocessing import Pool
from terminaltables import *

import sys
sys.path.append('./anet_toolkit/Evaluation')
from eval_detection import compute_average_precision_detection
from ops.utils import softmax
import os
import pickle
from ops.utils import get_configs


# options
parser = argparse.ArgumentParser(
    description="Evaluate detection performance metrics")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14'])
parser.add_argument('detection_pickles', type=str, nargs='+')
parser.add_argument('--nms_threshold', type=float, default=None)
parser.add_argument('--no_regression', default=False, action="store_true")
parser.add_argument('--softmax_before_filter', default=False, action="store_true")
parser.add_argument('-j', '--ap_workers', type=int, default=32)
parser.add_argument('--top_k', type=int, default=None)
parser.add_argument('--cls_scores', type=str, default=None)
parser.add_argument('--cls_top_k', type=int, default=1)
parser.add_argument('--score_weights', type=float, default=None, nargs='+')

args = parser.parse_args()

dataset_configs = get_configs(args.dataset)
num_class = dataset_configs['num_class']
test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])
nms_threshold = args.nms_threshold if args.nms_threshold else dataset_configs['evaluation']['nms_threshold']
top_k = args.top_k if args.top_k else dataset_configs['evaluation']['top_k']
softmax_bf = args.softmax_before_filter \
    if args.softmax_before_filter else dataset_configs['evaluation']['softmax_before_filter']

print("initiating evaluation of detection results {}".format(args.detection_pickles))
score_pickle_list = []
for pc in args.detection_pickles:
    score_pickle_list.append(pickle.load(open(pc, 'rb')))

if args.score_weights:
    weights = np.array(args.score_weights) / sum(args.score_weights)
else:
    weights = [1.0/len(score_pickle_list) for _ in score_pickle_list]


def merge_scores(vid):
    def merge_part(arrs, index, weights):
        if arrs[0][index] is not None:
            return np.sum([a[index] * w for a, w in zip(arrs, weights)], axis=0)
        else:
            return None

    arrays = [pc[vid] for pc in score_pickle_list]
    act_weights = weights
    comp_weights = weights
    reg_weights = weights
    rel_props = score_pickle_list[0][vid][0]

    return rel_props, \
           merge_part(arrays, 1, act_weights), \
           merge_part(arrays, 2, comp_weights), \
           merge_part(arrays, 3, reg_weights)

print('Merge detection scores from {} sources...'.format(len(score_pickle_list)))
detection_scores = {k: merge_scores(k) for k in score_pickle_list[0]}
print('Done.')

dataset = SSNDataSet("", test_prop_file, verbose=False)
dataset_detections = [dict() for i in range(num_class)]


if args.cls_scores:
    print('Using classifier scores from {}'.format(args.cls_scores))
    cls_score_pc = pickle.load(open(args.cls_scores, 'rb'), encoding='bytes')
    cls_score_dict = {os.path.splitext(os.path.basename(k.decode('utf-8')))[0]:v for k, v in cls_score_pc.items()}
else:
    cls_score_dict = None


# generate detection results
def gen_detection_results(video_id, score_tp):
    if len(score_tp[0].shape) == 3:
        rel_prop = np.squeeze(score_tp[0], 0)
    else:
        rel_prop = score_tp[0]

    # standardize regression scores
    reg_scores = score_tp[3]
    if reg_scores is None:
        reg_scores = np.zeros((len(rel_prop), num_class, 2), dtype=np.float32)
    reg_scores = reg_scores.reshape((-1, num_class, 2))

    if top_k <= 0 and cls_score_dict is None:
        combined_scores = softmax(score_tp[1])[:, 1:] * np.exp(score_tp[2])
        for i in range(num_class):
            loc_scores = reg_scores[:, i, 0][:, None]
            dur_scores = reg_scores[:, i, 1][:, None]
            try:
                dataset_detections[i][video_id] = np.concatenate((
                    rel_prop, combined_scores[:, i][:, None], loc_scores, dur_scores), axis=1)
            except:
                print(i, rel_prop.shape, combined_scores.shape, reg_scores.shape)
                raise
    elif cls_score_dict is None:
        combined_scores = softmax(score_tp[1][:, 1:]) * np.exp(score_tp[2])
        keep_idx = np.argsort(combined_scores.ravel())[-top_k:]
        for k in keep_idx:
            cls = k % num_class
            prop_idx = k // num_class
            if video_id not in dataset_detections[cls]:
                dataset_detections[cls][video_id] = np.array([
                    [rel_prop[prop_idx, 0], rel_prop[prop_idx, 1], combined_scores[prop_idx, cls],
                     reg_scores[prop_idx, cls, 0], reg_scores[prop_idx, cls, 1]]
                ])
            else:
                dataset_detections[cls][video_id] = np.vstack(
                    [dataset_detections[cls][video_id],
                     [rel_prop[prop_idx, 0], rel_prop[prop_idx, 1], combined_scores[prop_idx, cls],
                     reg_scores[prop_idx, cls, 0], reg_scores[prop_idx, cls, 1]]])
    else:
        if softmax_bf:
            combined_scores = softmax(score_tp[1])[:, 1:] * np.exp(score_tp[2])
        else:
            combined_scores = score_tp[1][:, 1:] * np.exp(score_tp[2])
        video_cls_score = cls_score_dict[os.path.splitext(os.path.basename(video_id))[0]]

        for video_cls in np.argsort(video_cls_score,)[-args.cls_top_k:]:
            loc_scores = reg_scores[:, video_cls, 0][:, None]
            dur_scores = reg_scores[:, video_cls, 1][:, None]
            try:
                dataset_detections[video_cls][video_id] = np.concatenate((
                    rel_prop, combined_scores[:, video_cls][:, None], loc_scores, dur_scores), axis=1)
            except:
                print(video_cls, rel_prop.shape, combined_scores.shape, reg_scores.shape, loc_scores.shape, dur_scores.shape)
                raise


print("Preprocessing detections...")
for k, v in detection_scores.items():
    gen_detection_results(k, v)
print('Done.')

# perform NMS
print("Performing nms...")
for cls in range(num_class):
    dataset_detections[cls] = {
        k: temporal_nms(v, nms_threshold) for k,v in dataset_detections[cls].items()
    }
print("NMS Done.")


def perform_regression(detections):
    t0 = detections[:, 0]
    t1 = detections[:, 1]
    center = (t0 + t1) / 2
    duration = (t1 - t0)

    new_center = center + duration * detections[:, 3]
    new_duration = duration * np.exp(detections[:, 4])

    new_detections = np.concatenate((
        np.clip(new_center - new_duration / 2, 0, 1)[:, None], np.clip(new_center + new_duration / 2, 0, 1)[:, None], detections[:, 2:]
    ), axis=1)
    return new_detections

# perform regression
if not args.no_regression:
    print("Performing location regression...")
    for cls in range(num_class):
        dataset_detections[cls] = {
            k: perform_regression(v) for k, v in dataset_detections[cls].items()
        }
    print("Regression Done.")
else:
    print("Skip regresssion as requested by --no_regression")


# ravel test detections
def ravel_detections(detection_db, cls):
    detection_list = []
    for vid, dets in detection_db[cls].items():
        detection_list.extend([[vid, cls] + x[:3] for x in dets.tolist()])
    df = pd.DataFrame(detection_list, columns=["video-id", "cls","t-start", "t-end", "score"])
    return df

plain_detections = [ravel_detections(dataset_detections, cls) for cls in range(num_class)]


# get gt
all_gt = pd.DataFrame(dataset.get_all_gt(), columns=["video-id", "cls","t-start", "t-end"])
gt_by_cls = []
for cls in range(num_class):
    gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop('cls', 1))

pickle.dump(gt_by_cls, open('gt_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(plain_detections, open('pred_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
print("Calling mean AP calculator from toolkit with {} workers...".format(args.ap_workers))

if args.dataset == 'activitynet1.2':
    iou_range = np.arange(0.5, 1.0, 0.05)
elif args.dataset == 'thumos14':
    iou_range = np.arange(0.1, 1.0, 0.1)
else:
    raise ValueError("unknown dataset {}".format(args.dataset))

ap_values = np.empty((num_class, len(iou_range)))


def eval_ap(iou, iou_idx, cls, gt, predition):
    # print(gt)
    # exit()
    ap = compute_average_precision_detection(gt, predition, iou)

    # print("class {} @ iou threshold {} AP: {}".format(cls, iou, ap))
    sys.stdout.flush()
    return cls, iou_idx, ap


def callback(rst):
    sys.stdout.flush()
    ap_values[rst[0], rst[1]] = rst[2][0]

pool = Pool(args.ap_workers)
jobs = []
for iou_idx, min_overlap in enumerate(iou_range):
    for cls in range(num_class):
        jobs.append(pool.apply_async(eval_ap, args=([min_overlap], iou_idx, cls, gt_by_cls[cls], plain_detections[cls]),
                               callback=callback))
        # print(eval_ap([min_overlap], iou_idx, cls, gt_by_cls[cls], plain_detections[cls]))
pool.close()
pool.join()
print("Evaluation done.\n\n")

map_iou = ap_values.mean(axis=0)
display_title = "Detection Performance on {}".format(args.dataset)

display_data = [["IoU thresh"], ["mean AP"]]

for i in range(len(iou_range)):
    display_data[0].append("{:.02f}".format(iou_range[i]))
    display_data[1].append("{:.04f}".format(map_iou[i]))

display_data[0].append('Average')
display_data[1].append("{:.04f}".format(map_iou.mean()))
table = AsciiTable(display_data, display_title)
table.justify_columns[-1] = 'right'
table.inner_footing_row_border = True
print(table.table)