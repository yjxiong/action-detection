import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from ssn_opts import parser
from load_binary_score import BinaryDataSet
from binary_model import BinaryClassifier
from transforms import *
from ops.utils import get_actionness_configs
from torch.utils import model_zoo
best_loss = 100

def main():
    global args, best_loss
    args = parser.parse_args()
    dataset_configs = get_actionness_configs(args.dataset)
    sampling_configs = dataset_configs['sampling']
    num_class = dataset_configs['num_class']
    args.dropout = 0.8
    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow','RGBDiff']:
        data_length = 5
    else:
        raise ValueError("unknown modality {}".format(args.modality))

    model = BinaryClassifier(num_class, args.num_body_segments,
                             args.modality, new_length = data_length,
                             base_model=args.arch, dropout=args.dropout,
                             bn_mode=args.bn_mode)

    if args.init_weights:
        if os.path.isfile(args.init_weights):
            print(("=> loading pretrained weights from '{}'".format(args.init_weights)))
            wd = torch.load(args.init_weights)
            model.base_model.load_state_dict(wd['state_dict'])
            print(("=> no weights file found at '{}'".format(args.init_weights)))
        else:
            print(("=> no weights file found at '{}'".format(args.init_weights)))
    elif args.kinetics_pretrain:
        model_url = dataset_configs['kinetics_pretrain'][args.arch][args.modality]
        model.base_model.load_state_dict(model_zoo.load_url(model_url)['state_dict'])
        print(("=> loaded init weights from '{}'".format(model_url)))
    else:
        # standard ImageNet pretraining
        if args.modality == 'Flow':
            model_url = dataset_configs['flow_init'][args.arch]
            model.base_model.load_state_dict(model_zoo.load_url(model_url)['state_dict'])
            print(("=> loaded flow init weights from '{}'".format(model_url)))


    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    cudnn.benchmark = True
    pin_memory = (args.modality == 'RGB')
    
    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()


    train_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['train_list'])
    val_prop_file = 'data/{}_proposal_list.txt'.format(dataset_configs['test_list'])
    train_loader = torch.utils.data.DataLoader(
        BinaryDataSet("", train_prop_file,
                      new_length=data_length,
                      modality=args.modality, exclude_empty=True,
                      body_seg=args.num_body_segments,
                      image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                      transform=torchvision.transforms.Compose([
                          train_augmentation,
                          Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                          ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                          normalize,
                      ])),
         batch_size=4, shuffle=True,
         num_workers=args.workers, pin_memory=pin_memory,
         drop_last = True) 


    val_loader = torch.utils.data.DataLoader(
         BinaryDataSet("", val_prop_file, new_length=data_length,
                       modality=args.modality, exclude_empty=True,
                       body_seg = args.num_body_segments,
                       image_tmpl="img_{:05}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                       random_shift=False, fg_ratio = 6, bg_ratio = 6,
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
         batch_size=4, shuffle=False,
         num_workers=args.workers, pin_memory=pin_memory)



    binary_criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))


    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        # train for one epoch
        train(train_loader, model, binary_criterion, optimizer, epoch)

        # evaluate on validation list
        if (epoch + 1) % args.eval_freq ==0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, binary_criterion, (epoch + 1) * len(train_loader))

        # remember best prec@1 and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to train model
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (out_frames, out_prop_type) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(out_frames)
        prop_type_var = torch.autograd.Variable(out_prop_type)

        # compute output

        binary_score, prop_type_target = model(input_var, prop_type_var)

        loss = criterion(binary_score, prop_type_target)

        losses.update(loss.data[0], out_frames.size(0))
        fg_acc = accuracy(binary_score.view(-1, 2, binary_score.size(1))[:,0,:].contiguous(),
                          prop_type_target.view(-1, 2)[:, 0].contiguous())
        bg_acc = accuracy(binary_score.view(-1, 2, binary_score.size(1))[:,1,:].contiguous(),
                          prop_type_target.view(-1, 2)[:, 1].contiguous())

        fg_accuracies.update(fg_acc[0].data[0], binary_score.size(0) // 2)
        bg_accuracies.update(bg_acc[0].data[0], binary_score.size(0) // 2)

        # compute gradient and do SGD step
        loss.backward()

        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print('Clipping gradient: {} with coef {}'.format(total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0
    
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '\n FG{fg_acc.val:.02f}({fg_acc.avg:.02f}) BG {bg_acc.val:.02f} ({bg_acc.avg:.02f})'
                  .format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, lr=optimizer.param_groups[0]['lr'],
                  fg_acc=fg_accuracies, bg_acc=bg_accuracies)
                 )


def validate(val_loader, model, criterion, iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    model.eval()

    end = time.time()

    for i, (out_frames, out_prop_type) in enumerate(val_loader):
        input_var = torch.autograd.Variable(out_frames, volatile=True)
        prop_type_var = torch.autograd.Variable(out_prop_type)

        # compute output
        binary_score, prop_type_target = model(input_var, prop_type_var)

        loss = criterion(binary_score, prop_type_target)
        losses.update(loss.data[0], out_frames.size(0))
        fg_acc = accuracy(binary_score.view(-1, 2, binary_score.size(1))[:,0,:].contiguous(),
                          prop_type_target.view(-1,2)[:, 0].contiguous())
        bg_acc = accuracy(binary_score.view(-1, 2, binary_score.size(1))[:,1,:].contiguous(),
                          prop_type_target.view(-1,2)[:, 1].contiguous())

        fg_accuracies.update(fg_acc[0].data[0], binary_score.size(0) // 2)
        bg_accuracies.update(bg_acc[0].data[0], binary_score.size(0) // 2)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'FG {fg_acc.val:.02f} BG {bg_acc.val:.02f}'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses,
                  fg_acc=fg_accuracies, bg_acc=bg_accuracies))

    print('Testing Results: Loss {loss.avg:.5f} \t'
          'FG Acc. {fg_acc.avg:.02f} BG Acc. {bg_acc.avg:.02f}'
          .format(loss=losses, fg_acc=fg_accuracies, bg_acc=bg_accuracies))

    return losses.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'binaryclassifier'+'_'.join((args.snapshot_pref, args.dataset, args.arch, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)




def adjust_learning_rate(optimizer, epoch, lr_steps):
    # Set the learning rate to the initial LR decayed by 10 every 30 epoches
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']





class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def accuracy(output, target, topk=(1,)):
    # computes the precision@k for the specific values of k
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




if __name__ == '__main__':
    main()
