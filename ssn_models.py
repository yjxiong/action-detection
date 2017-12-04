import torch
from torch import nn

from transforms import *
import torchvision.models

from ops.ssn_ops import Identity, StructuredTemporalPyramidPooling


class SSN(torch.nn.Module):
    def __init__(self, num_class,
                 starting_segment, course_segment, ending_segment, modality,
                 base_model='resnet101', new_length=None,
                 dropout=0.8,
                 crop_num=1, no_regression=False, test_mode=False,
                 stpp_cfg=(1, (1, 2), 1), bn_mode='frozen'):
        super(SSN, self).__init__()
        self.modality = modality
        self.num_segments = starting_segment + course_segment + ending_segment
        self.starting_segment = starting_segment
        self.course_segment = course_segment
        self.ending_segment = ending_segment
        self.reshape = True
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_regression = not no_regression
        self.test_mode = test_mode
        self.bn_mode=bn_mode

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
    Initializing SSN with base model: {}.
    SSN Configurations:
        input_modality:     {}
        starting_segments:  {}
        course_segments:    {}
        ending_segments:    {}
        num_segments:       {}
        new_length:         {}
        dropout_ratio:      {}
        loc. regression:    {}
        bn_mode:            {}
        
        stpp_configs:       {} 
            """.format(base_model, self.modality,
                       self.starting_segment, self.course_segment, self.ending_segment,
                       self.num_segments, self.new_length, self.dropout, 'ON' if self.with_regression else "OFF",
                       self.bn_mode, stpp_cfg)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_ssn(num_class, stpp_cfg)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.prepare_bn()

    def _prepare_ssn(self, num_class, stpp_cfg):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        self.stpp = StructuredTemporalPyramidPooling(feature_dim, True, configs=stpp_cfg)
        self.activity_fc = nn.Linear(self.stpp.activity_feat_dim(), num_class + 1)
        self.completeness_fc = nn.Linear(self.stpp.completeness_feat_dim(), num_class)

        nn.init.normal(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant(self.activity_fc.bias.data, 0)
        nn.init.normal(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant(self.completeness_fc.bias.data, 0)

        self.test_fc = None
        if self.with_regression:
            self.regressor_fc = nn.Linear(self.stpp.completeness_feat_dim(), 2 * num_class)
            nn.init.normal(self.regressor_fc.weight.data, 0, 0.001)
            nn.init.constant(self.regressor_fc.bias.data, 0)
        else:
            self.regressor_fc = None

        return feature_dim

    def prepare_bn(self):
        if self.bn_mode == 'partial':
            print("Freezing BatchNorm2D except the first one.")
            self.freeze_count = 2
        elif self.bn_mode == 'frozen':
            print("Freezing all BatchNorm2D layers")
            self.freeze_count = 1
        elif self.bn_mode == 'full':
            self.freeze_count = None
        else:
            raise ValueError("unknown bn mode")

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SSN, self).train(mode)
        count = 0
        if self.freeze_count is None:
            return

        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= self.freeze_count:
                    m.eval()

                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def prepare_test_fc(self):

        self.test_fc = nn.Linear(self.activity_fc.in_features,
                                 self.activity_fc.out_features
                                 + self.completeness_fc.out_features * self.stpp.feat_multiplier
                                 + (self.regressor_fc.out_features * self.stpp.feat_multiplier if self.with_regression else 0))
        reorg_comp_weight = self.completeness_fc.weight.data.view(
            self.completeness_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1)\
            .contiguous().view(-1, self.activity_fc.in_features)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(
            self.stpp.feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, reorg_comp_weight))
        bias = torch.cat((self.activity_fc.bias.data, reorg_comp_bias))

        if self.with_regression:
            reorg_reg_weight = self.regressor_fc.weight.data.view(
                self.regressor_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1) \
                .contiguous().view(-1, self.activity_fc.in_features)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                self.stpp.feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        linear_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                linear_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                # BN layers are all frozen in SSN
                bn_cnt += 1
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, aug_scaling, target, reg_target, prop_type):
        if not self.test_mode:
            return self.train_forward(input, aug_scaling, target, reg_target, prop_type)
        else:
            return self.test_forward(input)

    def train_forward(self, input, aug_scaling, target, reg_target, prop_type):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        activity_ft, completeness_ft = self.stpp(base_out, aug_scaling, [self.starting_segment,
                                                                         self.starting_segment + self.course_segment,
                                                                         self.num_segments])

        raw_act_fc = self.activity_fc(activity_ft)
        raw_comp_fc = self.completeness_fc(completeness_ft)

        type_data = prop_type.view(-1).data
        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()
        comp_indexer = ((type_data == 0) + (type_data == 1)).nonzero().squeeze()
        target = target.view(-1)

        if self.with_regression:
            reg_target = reg_target.view(-1, 2)
            reg_indexer = (type_data == 0).nonzero().squeeze()
            raw_regress_fc = self.regressor_fc(completeness_ft).view(-1, self.completeness_fc.out_features, 2)
            return raw_act_fc[act_indexer, :], target[act_indexer], \
                   raw_comp_fc[comp_indexer, :], target[comp_indexer], \
                   raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :]
        else:
            return raw_act_fc[act_indexer, :], target[act_indexer], \
                   raw_comp_fc[comp_indexer, :], target[comp_indexer]

    def test_forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        return self.test_fc(base_out), base_out

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
