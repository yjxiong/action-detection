import numpy as np
import glob
import os

def flow_stack_oversample(flow_stack, crop_dims):
    """
    This function performs oversampling on flow stacks.
    Adapted from pyCaffe's oversample function
    :param flow_stack:
    :param crop_dims:
    :return:
    """
    im_shape = np.array(flow_stack.shape[1:])
    stack_depth = flow_stack.shape[0]
    crop_dims = np.array(crop_dims)

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])

    h_center_offset = (im_shape[0] - crop_dims[0])/2
    w_center_offset = (im_shape[1] - crop_dims[1])/2

    crop_ix = np.empty((5, 4), dtype=int)

    cnt = 0
    for i in h_indices:
        for j in w_indices:
            crop_ix[cnt, :] = (i, j, i+crop_dims[0], j+crop_dims[1])
            cnt += 1
    crop_ix[4, :] = [h_center_offset, w_center_offset,
                     h_center_offset+crop_dims[0], w_center_offset+crop_dims[1]]

    crop_ix = np.tile(crop_ix, (2,1))

    crops = np.empty((10, flow_stack.shape[0], crop_dims[0], crop_dims[1]),
                     dtype=flow_stack.dtype)

    for ix in xrange(10):
        cp = crop_ix[ix]
        crops[ix] = flow_stack[:, cp[0]:cp[2], cp[1]:cp[3]]
    crops[5:] = crops[5:, :, :, ::-1]
    crops[5:, range(0, stack_depth, 2), ...] = 255 - crops[5:, range(0, stack_depth, 2), ...]
    return crops


def rgb_oversample(image, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Adapted from Caffe
    Parameters
    ----------
    image : (H x W x K) ndarray
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10 x H x W x K) ndarray of crops.
    """
    # Dimensions and center.
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 , crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)

    ix = 0
    for crop in crops_ix:
        crops[ix] = image[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def rgb_to_parrots(frame, oversample=True, mean_val=None, crop_size=None, rescale=False):
    """
    Pre-process the rgb frame for Parrots input
    """
    if mean_val is None:
        mean_val = [104, 117, 123]
    if not oversample:
        ret_frame = (frame - mean_val).transpose((2, 0, 1))
        if rescale:
            ret_frame /= 57.3
        return ret_frame[None, ...]
    else:
        crops = rgb_oversample(frame, crop_size) - mean_val
        ret_frames = crops.transpose((0, 3, 1, 2))
        if rescale:
            ret_frames /= 57.3
        return ret_frames


def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in xrange(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data


def load_proposal_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(info):
        offset = 0
        vid = info[offset][-11:]
        offset += 1

        n_frame = int(float(info[1]) * float(info[2]))
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset+n_gt]]
        offset += n_gt

        n_pr = int(info[offset])
        offset += 1
        pr_boxes = [x.split() for x in info[offset:offset+n_pr]]

        return vid, n_frame, gt_boxes, pr_boxes

    return [parse_group(l) for l in info_list]


def dump_window_list(video_info, named_proposals, frame_path, name_pattern, allow_empty=False, score=None):

    # first count frame numbers
    try:
        video_name = video_info.path.split('/')[-1].split('.')[0]
        files = glob.glob(os.path.join(frame_path, video_name, name_pattern))
        frame_cnt = len(files)
    except:
        if allow_empty:
            frame_cnt = score.shape[0] * 6
            video_name = video_info.id
        else:
            raise

    # convert time to frame number
    real_fps = float(frame_cnt) / video_info.duration

    # get groundtruth windows
    gt_w = [(x.num_label, x.time_span) for x in video_info.instance]
    gt_windows = [(x[0]+1, int(x[1][0] * real_fps), int(x[1][1] * real_fps)) for x in gt_w]

    dump_gt = []
    for gt in gt_windows:
        dump_gt.append('{} {} {}'.format(*gt))

    dump_proposals = []
    for pr in named_proposals:
        real_start = int(pr[3] * real_fps)
        real_end = int(pr[4] * real_fps)
        label = pr[0]
        overlap = pr[1]
        overlap_self = pr[2]
        dump_proposals.append('{} {:.04f} {:.04f} {} {}'.format(label, overlap, overlap_self, real_start, real_end))

    ret_str = '{path}\n{duration}\n{fps}\n{num_gt}\n{gts}\n{num_window}\n{prs}\n'.format(
        path=os.path.join(frame_path, video_name), duration=frame_cnt, fps=1,
        num_gt=len(dump_gt), gts='\n'.join(dump_gt),
        num_window=len(dump_proposals), prs='\n'.join(dump_proposals))

    return ret_str


def split_proposal_file(filename):
    basics_dict, gt_dict, proposal_dict = {}, {}, {}
    with open(filename) as f:
        while True:
            if f.readline().startswith('#'):
                video_path = f.readline().strip()
                vid = video_path.split('/')[-1]
                video_duration = float(f.readline().strip())
                video_fps = float(f.readline().strip())
                num_gt_windows = int(f.readline().strip())
                basics_dict[vid] = (video_duration, video_fps)
                video_gt_windows = []
                for i in xrange(num_gt_windows):
                      [label, start_time, end_time] = f.readline().strip().split(' ')
                      label = int(label)
                      start_time = float(start_time)
                      end_time = float(end_time)
                      video_gt_windows.append((label, start_time, end_time))
                gt_dict[vid] = video_gt_windows
                if num_gt_windows==0:
                    gt_dict[vid] = []
                    f.readline()
                num_proposal_windows = int(f.readline().strip())
                video_proposal_windows = []
                if num_proposal_windows==0:
                    f.readline()
                for i in xrange(num_proposal_windows):
                    [label, overlap, overlap_with_self, start_time, end_time] = f.readline().strip().split(' ')
                    label = int(label)
                    overlap = float(overlap)
                    overlap_with_self = float(overlap_with_self)
                    start_time = float(start_time)
                    end_time = float(end_time)
                    video_proposal_windows.append((label, overlap, overlap_with_self, start_time, end_time))
                proposal_dict[vid] = video_proposal_windows
            else:
                break
    return basics_dict, gt_dict, proposal_dict
