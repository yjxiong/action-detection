import argparse
import time

import numpy as np

from ssn_dataset import SSNDataSet
from ssn_models import SSN
from transforms import *
from ops.ssn_ops import STPPReorgainzed
from torch import multiprocessing
from torch.utils import model_zoo
from ops.utils import get_configs, get_reference_model_url


parser = argparse.ArgumentParser(
    description="SSN Testing Tool")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--aug_ratio', type=float, default=0.5)
parser.add_argument('--frame_interval', type=int, default=6)
parser.add_argument('--test_batchsize', type=int, default=512)
parser.add_argument('--no_regression', action="store_true", default=False)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_pref', type=str, default='')
parser.add_argument('--use_reference', default=False, action='store_true')
parser.add_argument('--use_kinetics_reference', default=False, action='store_true')

args = parser.parse_args()

dataset_configs = get_configs(args.dataset)

num_class = dataset_configs['num_class']
stpp_configs = tuple(dataset_configs['stpp'])
test_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])

if args.modality == 'RGB':
    data_length = 1
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5
else:
    raise ValueError("unknown modality {}".format(args.modality))

gpu_list = args.gpus if args.gpus is not None else range(8)


def runner_func(dataset, state_dict, stats, gpu_id, index_queue, result_queue):
    torch.cuda.set_device(gpu_id)
    net = SSN(num_class, 2, 5, 2,
              args.modality, test_mode=True,
              base_model=args.arch, no_regression=args.no_regression, stpp_cfg=stpp_configs)
    net.load_state_dict(state_dict)
    net.prepare_test_fc()
    net.eval()
    net.cuda()
    output_dim = net.test_fc.out_features
    reorg_stpp = STPPReorgainzed(output_dim, num_class + 1, num_class,
                                 num_class * 2, True, stpp_cfg=stpp_configs)

    while True:
        index = index_queue.get()
        frames_gen, frame_cnt, rel_props, prop_ticks, prop_scaling = dataset[index]
        num_crop = args.test_crops
        length = 3
        if args.modality == 'Flow':
            length = 10
        elif args.modality == 'RGBDiff':
            length = 18

        output = torch.zeros((frame_cnt, output_dim)).cuda()
        cnt = 0
        for frames in frames_gen:
            input_var = torch.autograd.Variable(frames.view(-1, length, frames.size(-2), frames.size(-1)).cuda(),
                                                volatile=True)
            rst, _ = net(input_var, None, None, None, None)
            sc = rst.data.view(num_crop, -1, output_dim).mean(dim=0)
            output[cnt: cnt + sc.size(0), :] = sc
            cnt += sc.size(0)
        act_scores, comp_scores, reg_scores = reorg_stpp.forward(output, prop_ticks, prop_scaling)

        if reg_scores is not None:
            reg_scores = reg_scores.view(-1, num_class, 2)
            reg_scores[:, :, 0] = reg_scores[:, :, 0] * stats[1, 0] + stats[0, 0]
            reg_scores[:, :, 1] = reg_scores[:, :, 1] * stats[1, 1] + stats[0, 1]

        # perform stpp on scores
        result_queue.put((dataset.video_list[index].id, rel_props.numpy(), act_scores.cpu().numpy(), \
               comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), output.cpu().numpy()))


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')  # this is crucial to using multiprocessing processes with PyTorch

    # This net is used to provides setup settings. It is not used for testing.
    net = SSN(num_class, 2, 5, 2,
              args.modality, test_mode=True,
              base_model=args.arch, no_regression=args.no_regression, stpp_cfg=stpp_configs)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    if not args.use_reference and not args.use_kinetics_reference:
        checkpoint = torch.load(args.weights)
    else:
        model_url = get_reference_model_url(args.dataset, args.modality,
                                            'ImageNet' if args.use_reference else 'Kinetics', args.arch)
        checkpoint = model_zoo.load_url(model_url)
        print("using reference model: {}".format(model_url))

    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    stats = checkpoint['reg_stats'].numpy()

    dataset = SSNDataSet("", test_prop_file,
                         new_length=data_length,
                         modality=args.modality,
                         aug_seg=2, body_seg=5,
                         image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB",
                                                                          "RGBDiff"] else args.flow_pref + "{}_{:05d}.jpg",
                         test_mode=True, test_interval=args.frame_interval,
                         transform=torchvision.transforms.Compose([
                             cropping,
                             Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                             ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                             GroupNormalize(net.input_mean, net.input_std),
                         ]), reg_stats=stats, verbose=False)

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func, args=(dataset, base_dict, stats, gpu_list[i % len(gpu_list)], index_queue, result_queue))
               for i in range(args.workers)]

    del net

    for w in workers:
        w.daemon = True
        w.start()

    max_num = args.max_num if args.max_num > 0 else len(dataset)

    for i in range(max_num):
        index_queue.put(i)

    proc_start_time = time.time()
    out_dict = {}
    for i in range(max_num):
        rst = result_queue.get()
        out_dict[rst[0]] = rst[1:]
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {:.04f} sec/video'.format(i, i + 1,
                                                                        max_num,
                                                                        float(cnt_time) / (i + 1)))

    if args.save_scores is not None:
        save_dict = {k: v[:-1] for k,v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_scores, 'wb'), pickle.HIGHEST_PROTOCOL)

    if args.save_raw_scores is not None:
        save_dict = {k: v[-1] for k,v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_raw_scores, 'wb'), pickle.HIGHEST_PROTOCOL)
