import torch
from torch import nn

from transforms import *
import torchvision.models

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, modality,
                 base_model='resnet101', new_length=None,
                 dropout=0.8,
                 crop_num=1, test_mode=False, bn_mode='frozen'):

        super(BinaryClassifier, self).__init__()
        self.modality = modality
        self.num_segments = course_segment
        self.course_segment = course_segment
        self.reshape = True
        self.dropout = dropout
        self.crop_num = crop_num
        self.test_mode = test_mode
        self.bn_mode = bn_mode

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        print(("""
               Initializing BinaryClassifier with base model:{}
               BinaryClassifier Configurations:
                   input_modality: {}
                   course_segment: {}
                   num_segments:   {}
                   new_length:     {}
                   dropout_ratio:  {}
                   bn_mode:        {}
              """.format(base_model, self.modality, self.course_segment, self.num_segments,
                         self.new_length, self.dropout, self.bn_mode)))

        self._prepare_base_model(base_model)
        
        feature_dim = self._prepare_binary_classifier(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model) 
            print("Done. Flow model readly...")      
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self.construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
 
        self.prepare_bn()


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchincal way
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim = 1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if necessary
        layer_name = list(container.state_dict().keys())[0][:-7] #remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model


    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convoltion layers
        # Torch models are usually defined in a hierarchical way.
        # nn.moduls.children(0 return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if keep_rgb:
            new_kernel_size = kernel.size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data.mean(dim=1).expand(new_kernel_size).contiguous()),1)
            new_kernel_size = kernel_size[:1] + (3 + 3*self.new_length,) + kernel_size[2:]
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels, 
                             conv_layer.kernel_size, conv.layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if necessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model



    def _prepare_binary_classifier(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        self.classifier_fc = nn.Linear(feature_dim, num_class)

        nn.init.normal(self.classifier_fc.weight.data, 0, 0.001)
        nn.init.constant(self.classifier_fc.bias.data, 0)

        self.test_fc = None
        self.feature_dim = feature_dim

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
        super(BinaryClassifier, self).train(mode)
        count = 0
        if self.freeze_count is None:
            return

        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= self.freeze_count:
                    m.eval()

                    # shutdown update in frozen mode
                    m.weight_requires_grad = False
                    m.bias.requires_grad = False



    def forward(self, inputdata, target):
        if not self.test_mode:
            return self.train_forward(inputdata, target)
        else:
            return self.test_forward(inputdata)


    def train_forward(self, inputdata, target):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        base_out = self.base_model(inputdata.view((-1, sample_len) + inputdata.size()[-2:]))
        src = base_out.view(-1, self.course_segment, base_out.size()[1])
        course_ft = src[:, :, :].mean(dim=1)
        raw_course_ft = self.classifier_fc(course_ft)
        target = target.view(-1)

        return raw_course_ft, target
                

    def test_forward(self, input):
        sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
        base_out = self.base_model(input.view((-1,sample_len) + input.size()[-2:]))
        return self.test_fc(base_out), base_out




    def prepare_test_fc(self):

        self.test_fc = nn.Linear(self.classifier_fc.in_features,
                                 self.classifier_fc.out_features)

        weight = self.classifier_fc.weight.data
        bias = self.classifier_fc.bias.data

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
                # BN layers are all frozen
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
                                                   GroupRandomHorizontalFlip(is_flow=Flase)])
