import argparse
import os
import sys
import math
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import time
import pickle
import multiprocessing as mp
from ops.sequence_funcs import *
from ops.anet_db import ANetDB
from ops.thumos_db import THUMOSDB 
from ops.detection_metrics import get_temporal_proposal_recall, name_proposal
from ops.sequence_funcs import temporal_nms
from ops.io import dump_window_list
parser = argparse.ArgumentParser()
parser.add_argument('score_files', type=str, nargs='+')
parser.add_argument("--anet_version", type=str, default='1.2', help='')
parser.add_argument("--dataset", type=str, default='activitynet', choices=['activitynet', 'thumos14'])
parser.add_argument("--cls_scores", type=str, default=None,
                    help='classification scores, if set to None, will use groundtruth labels')
parser.add_argument("--subset", type=str, default='validation', choices=['training', 'validation', 'testing'])
parser.add_argument("--iou_thresh", type=float, nargs='+', default=[0.5, 0.75, 0.95])
parser.add_argument("--score_weights", type=float, nargs='+', default=None, help='')
parser.add_argument("--write_proposals", type=str, default=None, help='')
parser.add_argument("--minimum_len", type=float, default=0, help='minimum length of a proposal, in second')
parser.add_argument("--reg_score_files", type=str, nargs='+', default=None)
parser.add_argument("--frame_path", type=str, default='/mnt/SSD/ActivityNet/anet_v1.2_extracted_340/')

args = parser.parse_args()


if args.dataset == 'activitynet':
    db = ANetDB.get_db(args.anet_version)
    db.try_load_file_path('/mnt/SSD/ActivityNet/anet_v1.2_extracted_340/')
elif args.dataset == 'thumos14':
    db = THUMOSDB.get_db()
    db.try_load_file_path('/mnt/SSD/THUMOS14/')

    # rename subset test
    if args.subset == 'testing':
        args.subset = 'test'
else:
    raise ValueError("unknown dataset {}".format(args.dataset))

video_list = db.get_subset_videos(args.subset)
video_list = [v for v in video_list if v.instances != []]
print("video list size: {}".format(len(video_list)))
# load scores
print('loading scores...')
score_list = []
for fname in args.score_files:
    score_list.append(pickle.load(open(fname, 'rb')))
print('load {} piles of scores'.format(len(score_list)))


# load classification scores if specified
if args.cls_scores:
    cls_scores = cPickle.load(open(args.cls_scores, 'rb'))
else:
    cls_scores = None
print('done')

# load regression scores
if args.reg_score_files is not None:
    print('loading regression scores')
    reg_score_list = []
    for fname in args.reg_score_files:
        reg_score_list.append(cPickle.load(open(fname, 'rb')))
    print('load {} piles of regression scores'.format(len(reg_score_list)))
else:
    reg_score_list = None


# merge scores
print('merging scores')
score_dict = {}
for key in score_list[0].keys():
    out_score = score_list[0][key].mean(axis=1) * (1.0 if args.score_weights is None else args.score_weights[0])
    for i in range(1, len(score_list)):
        add_score = score_list[i][key].mean(axis=1)
        if add_score.shape[0] < out_score.shape[0]:
            out_score = out_score[:add_score.shape[0], :]
        elif add_score.shape[0] > out_score.shape[0]:
            tick = add_score.shape[0] / float(out_score.shape[0])
            indices = [int(x * tick) for x in range(out_score.shape[0])]
            add_score = add_score[indices, :]
        out_score += add_score * (1.0 if args.score_weights is None else args.score_weights[i])
    score_dict[key] = out_score
print('done')

# merge regression scores
if reg_score_list is not None:
    print('merging regression scores')
    reg_score_dict = {}
    for key in reg_score_list[0].keys():
        out_score = reg_score_list[0][key].mean(axis=1)
        for i in range(1, len(reg_score_list)):
            add_score = reg_score_list[i][key].mean(axis=1)
            if add_score.shape[0] < out_score.shape[0]:
                out_score = out_score[:add_score.shape[0], :]
            out_score += add_score
        reg_score_dict[key] = out_score / len(reg_score_list)
    print('done')
else:
    reg_score_dict = None

# bottom-up generate proposals
print('generating proposals')
pr_dict = {}
pr_score_dict = {}
topk = 1


def gen_prop(v):
    if (args.dataset == 'activitynet') or (args.dataset == 'thumos14'):
        vid = v.id
    else:
        vid = v.path.split('/')[-1].split('.')[0]
    scores = score_dict[vid]
    frm_duration = len(scores)
    topk_cls = [0]
    topk_labels = label_frame_by_threshold(scores, topk_cls, bw=3, thresh=[0.01, 0.05, 0.1, .15, 0.25, .4, .5, .6, .7, .8, .9, .95, ], multicrop=False)

    bboxes = []
    tol_lst = [0.05, .1, .2, .3, .4, .5, .6, 0.8, 1.0]

    bboxes.extend(build_box_by_search(topk_labels, np.array(tol_lst)))
    if reg_score_dict:
        reg_scores = reg_score_dict[score_id]
        bboxes = regress_box(bboxes, reg_scores, len(scores))


    # print len(bboxes)
    bboxes = temporal_nms(bboxes, 0.9)

    pr_box = [(x[0] / float(frm_duration) * v.duration, x[1] / float(frm_duration) * v.duration) for x in bboxes]

    # filter out too short proposals
    pr_box = list(filter(lambda b: b[1] - b[0] > args.minimum_len, pr_box))
    return v.id, pr_box, [x[3] for x in bboxes]


def call_back(rst):
    pr_dict[rst[0]] = rst[1]
    pr_score_dict[rst[0]] = rst[2]
    import sys
    print(rst[0], len(pr_dict), len(rst[1]))
    sys.stdout.flush()

pool = mp.Pool(processes = 32)
lst = []
handle = [pool.apply_async(gen_prop, args=(x, ), callback=call_back) for x in video_list]
pool.close()
pool.join()

# evaluate proposal info
proposal_list = [pr_dict[v.id] for v in video_list if v.id in pr_dict]
gt_spans_full = [[(x.num_label, x.time_span) for x in v.instances] for v in video_list if v.id in pr_dict]
gt_spans = [[item[1] for item in x] for x in gt_spans_full]
score_list = [score_dict[v.id] for v in video_list if v.id in pr_dict]
duration_list = [v.duration for v in video_list if v.id in pr_dict]
proposal_score_list = [pr_score_dict[v.id] for v in video_list if v.id in pr_dict]
print('{} groundtruth boxes from'.format(sum(map(len, gt_spans))))



print('average # of proposals: {}'.format(np.mean(list(map(len, proposal_list)))))
IOU_thresh = np.arange(0.5, 1, 0.2)
p_list = []
for th in IOU_thresh:
    pv, pi = get_temporal_proposal_recall(proposal_list, gt_spans, th)
    print('IOU threshold {}. per video recall: {:02f}, per instance recall: {:02f}'.format(th, pv * 100, pi * 100))
    p_list.append((pv, pi))
print('Average Recall: {:.04f} {:.04f}'.format(*(np.mean(p_list, axis=0)*100)))

if args.write_proposals:

    name_pattern = 'img_*.jpg'
    frame_path = args.frame_path

    named_proposal_list = [name_proposal(x, y) for x, y in zip(gt_spans_full, proposal_list)]
    allow_empty = args.dataset == 'activitynet' and args.subset == 'testing'
    dumped_list = [dump_window_list(v, prs, frame_path, name_pattern, score=score, allow_empty=allow_empty) for v, prs, score in
                   zip(filter(lambda x: x.id in pr_dict, video_list), named_proposal_list, score_list)]

    with open(args.write_proposals, 'w') as of:
        for i, e in enumerate(dumped_list):
            of.write('# {}\n'.format(i + 1))
            of.write(e)

    print('list written. got {} videos'.format(len(dumped_list)))
