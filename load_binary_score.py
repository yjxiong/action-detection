import torch.utils.data as data

import os
import os.path
from numpy.random import randint
from ops.io import load_proposal_file
from transforms import *
from ops.utils import temporal_iou



class BinaryInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None, overlap_self=None, iou=1.0):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps
        self.iou = iou
        self.coverage = (end_frame - start_frame) / video_frame_count

        self.overlap_self = overlap_self

    @property
    def start_time(self):
        return self.start_frame / self.fps
    
    @property
    def label(self):
        return self._lable if self._label is not None else -1






class BinaryVideoRecord:
    def __init__(self, prop_record):
        self._data = prop_record
#        print(prop_record)

        frame_count = int(self._data[1])
        
        # build instance record
        self.fps = 1
        self.gt = [
            BinaryInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), iou = 1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]
        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))
        
        self.proposals = [
            BinaryInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]), iou = float(x[1]),
                           overlap_self = float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(filter(lambda x:x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True):
        fg = [p for p in self.proposals if p.iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)
        return fg

    def get_bg(self, bg_thresh):
        bg = [p for p in self.proposals if p.iou < bg_thresh]
        return bg



class BinaryDataSet(data.Dataset):

    def __init__(self, root_path,
                 prop_file = None,
                 body_seg=5, video_centric=True,
                 new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 prop_per_video=12, fg_ratio=3, bg_ratio=9,
                 fg_iou_thresh=0.7,
                 bg_iou_thresh=0.01,
                 bg_coverage_thresh=0.02,
                 gt_as_fg=True, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1):

        self.root_path = root_path
        self.prop_file = prop_file
        self.verbose = verbose

        self.body_seg = body_seg
        self.video_centric = video_centric
        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier

        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.test_interval = test_interval

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.bg_coverage_thresh = bg_coverage_thresh
        self.starting_ratio = 0.5
        self.ending_ratio = 0.5

        self.gt_as_fg = gt_as_fg
        denum = fg_ratio + bg_ratio

        self.fg_per_video = int(prop_per_video * (fg_ratio / denum))
        self.bg_per_video = int(prop_per_video * (bg_ratio / denum))

        self._parse_prop_file()

    def _parse_prop_file(self):
        prop_info = load_proposal_file(self.prop_file)
        
        self.video_list = [BinaryVideoRecord(p) for p in prop_info]

        if self.exclude_empty:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        self.video_dict = {v.id: v for v in self.video_list}

        # construct two pools:
        # 1. Foreground
        # 2. Background

        self.fg_pool = []
        self.bg_pool = []

        for v in self.video_list:
            self.fg_pool.extend([(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)])
            self.bg_pool.extend([(v.id, prop) for prop in v.get_bg(self.bg_iou_thresh)])

        if self.verbose:
            print("""
            
            BinaryDataSet: Proposal file {prop_file} parse.

            There are {pnum} usable proposals from {vnum} videos.
            {fnum} foreground proposals
            {bnum} background proposals

            Sampling config:
            FG/BG: {fr}/{br}
            
            Epoch size muiltiplier: {em}
            """.format(prop_file=self.prop_file, pnum=len(self.fg_pool) + len(self.bg_pool),
                       fnum=len(self.fg_pool), bnum=len(self.bg_pool),
                       fr = self.fg_per_video, br=self.bg_per_video, vnum=len(self.video_dict),
                       em=self.epoch_multiplier))
        else:
            print("""
                       BinaryDataset: proposal file {prop_file} parsed.
            """.format(prop_file=self.prop_file))
      #  return self.video_list


    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index], self.test_interval)
        else:
            return self.get_training_data(real_index)

    def _sample_frames(self, prop):
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame
        duration = end_frame - start_frame + 1
        sample_duration = duration / self.body_seg        

        if sample_duration < 1:
            return start_frame + randint(prop.end_frame - prop.start_frame, size = self.body_seg)

        frame_indice = []
        split_stage = [int(np.round(i*sample_duration)) + start_frame for i in range(self.body_seg+1) ]
        
        for i in range(self.body_seg):
           #  print(split_stage[i], split_stage[i+1])
            index = np.random.choice(range(split_stage[i],split_stage[i+1]), 1)
            frame_indice.extend(index)
        return frame_indice
  

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]
        

    def _load_prop_data(self, prop):

        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames
        # frame_cnt = 1572 
        frame_selected = self._sample_frames(prop[0][1])
        frames = []
        for idx in frame_selected:
            for x in range(self.new_length):
                frames.extend(self._load_image(prop[0][0], min(frame_cnt, idx+x)))

        return frames, prop[1]
        # sample segment indices


    def _video_centric_sampling(self, video):
        
        fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg)
        bg = video.get_bg(self.bg_iou_thresh)
        
        def sample_video_proposals(proposal_type, video_id, video_pool, requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is noting in the video pool, go fetch from the dataset pool
                return [(dataset_pool[x], proposal_type) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
            else:
                replicate = len(video_pool) < requested_num
                idx = np.random.choice(len(video_pool), requested_num, replace = replicate)
                return [((video_id, video_pool[x]), proposal_type) for x in idx]

        out_props = []
        out_props.extend(sample_video_proposals(1, video.id, fg, self.fg_per_video, self.fg_pool)) # sample foreground
        out_props.extend(sample_video_proposals(0, video.id, bg, self.bg_per_video, self.bg_pool)) # sample background
        
        return out_props


    def get_training_data(self, index):

        video = self.video_list[index]
        props = self._video_centric_sampling(video)
        
        out_frames = []
        out_prop_len = []
        out_prop_type = []
         
        frames = []
        for idx, p in enumerate(props):
            prop_frames, prop_type = self._load_prop_data(p)
            processed_frames = self.transform(prop_frames)
            out_frames.append(processed_frames)
            out_prop_type.append(prop_type)

        out_prop_type = torch.from_numpy(np.array(out_prop_type))
        out_frames = torch.cat(out_frames)
        return out_frames, out_prop_type


    def get_test_data(self, video, test_interval, gen_batchsize=4):
        props = []
        video_id = video.id
        frame_cnt = video.num_frames

        frame_ticks = np.arange(0, frame_cnt - self.new_length, test_interval, dtype=np.int) + 1
        num_sampled_frames = len(frame_ticks)

        # avoid empty proposal list
        for i in frame_ticks:
            props.append(BinaryInstance(i, i+1, 1))

        proposal_tick_list = []

        for proposal in props:
            proposal_ticks = proposal.start_frame, proposal.end_frame
            proposal_tick_list.append(proposal_ticks)

        # load frames
        # Since there are many frames for each video during testing, instead of returning the read frames
        # we return a generator which gives the frames in samll batches, this lower the momeory burden
        # runtime overhead. Usually stting batchsize=4 would fit most cases.

        def frame_gen(batchsize):
            frames= []
            cnt = 0
            for idx, seg_ind in enumerate(frame_ticks):
                p = int(seg_ind)
                for x in range(self.new_length):
                    frames.extend(self._load_image(video_id, min(frame_cnt, p+x)))
                cnt += 1

                if cnt % batchsize == 0:
                    frames = self.transform(frames)
                    yield frames
                    frames = []
            
            if len(frames):
                frames = self.transform(frames)
                yield frames

        return frame_gen(gen_batchsize), len(frame_ticks)



    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier 
