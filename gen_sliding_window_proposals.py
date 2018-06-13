from ops.anet_db import ANetDB
from ops.thumos_db import THUMOSDB
import numpy as np
import multiprocessing
import argparse
from ops.detection_metrics import get_temporal_proposal_recall, name_proposal
from ops.sequence_funcs import gen_exponential_sw_proposal
from ops.io import dump_window_list


parser = argparse.ArgumentParser(description="Make window file used for detection")
parser.add_argument("subset")
parser.add_argument("modality", choices=['rgb', 'flow'])
parser.add_argument("frame_path")
parser.add_argument("output_file")
parser.add_argument("--overlap", type=float, default=0.7)
parser.add_argument("--max_level", type=int, default=8)
parser.add_argument("--time_step", type=float, default=1)
parser.add_argument("--version",  default="1.2")
parser.add_argument("--avoid",  default=None, type=str)
parser.add_argument("--dataset",  default="activitynet", choices=['thumos14', 'activitynet'])
args = parser.parse_args()

name_pattern = 'img_*.jpg' if args.modality == 'rgb' else 'flow_x_*.jpg'

if args.dataset == 'activitynet':
    db = ANetDB.get_db(args.version)
    db.try_load_file_path(args.frame_path)
elif args.dataset == 'thumos14':
    db = THUMOSDB.get_db()
    db.try_load_file_path(args.frame_path)

    if args.subset == 'testing':
        args.subset = 'test'

else:
    raise ValueError("Unknown dataset {}".format(args.dataset))

avoid_list = [x.strip() for x in open(args.avoid)] if args.avoid else []


videos = db.get_subset_videos(args.subset)

# generate proposals and name them
gt_spans = [[(x.num_label, x.time_span) for x in v.instances] for v in videos]
proposal_list = list(map(lambda x: gen_exponential_sw_proposal(x,
                                                          overlap=args.overlap,
                                                          time_step=args.time_step,
                                                          max_level=args.max_level), videos))
print('average # of proposals: {} at overlap param {}'.format(np.mean(list(map(len, proposal_list))), args.overlap))

named_proposal_list = [name_proposal(x, y) for x,y in zip(gt_spans, proposal_list)]
recall_list = []
IOU_thresh = [0.5, 0.7, 0.9]
for th in IOU_thresh:
    pv, pi = get_temporal_proposal_recall(proposal_list, [[y[1] for y in x] for x in gt_spans], th)
    print('IOU threshold {}. per video recall: {:02f}, per instance recall: {:02f}'.format(th, pv * 100, pi * 100))
    recall_list.append([args.overlap, th, np.mean(list(map(len, proposal_list))), pv, pi])
print("average per video recall: {:.2f}, average per instance recall: {:.2f}".format(
    np.mean([x[3] for x in recall_list]), np.mean([x[4] for x in recall_list])))

dumped_list = [dump_window_list(v, prs, args.frame_path, name_pattern) for v, prs in zip(videos, named_proposal_list) if v.id not in avoid_list]

with open(args.output_file, 'w') as of:
    for i, e in enumerate(dumped_list):
        of.write('# {}\n'.format(i + 1))
        of.write(e)

print('list written. got {} videos'.format(len(dumped_list)))
