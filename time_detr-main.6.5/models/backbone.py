# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
我们更换自己的backbone部分
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # layer0 layer1不需要训练 因为前面层提取的信息其实很有限 都是差不多的 不需要训练
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # False 检测任务不需要返回中间层
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # 检测任务直接返回layer4即可  执行torchvision.models._utils.IntermediateLayerGetter这个函数可以直接返回对应层的输出结果
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        """
                tensor_list: pad预处理之后的图像信息
                tensor_list.tensors: [bs, 3, 608, 810]预处理后的图片数据 对于小图片而言多余部分用0填充
                tensor_list.mask: [bs, 608, 810] 用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
                """
        # 取出预处理后的图片数据 [bs, 3, 608, 810] 输入模型中  输出layer4的输出结果 dict '0'=[bs, 2048, 19, 26]
        xs = self.body(tensor_list.tensors)
        # 保存输出数据
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask    # 取出图片的mask [bs, 608, 810] 知道图片哪些区域是有效的 哪些位置是pad之后的无效的
            assert m is not None
            # 通过插值函数知道卷积后的特征的mask  知道卷积后的特征哪些是有效的  哪些是无效的
            # 因为之前图片输入网络是整个图片都卷积计算的 生成的新特征其中有很多区域都是无效的
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # out['0'] = NestedTensor: tensors[bs, 2048, 19, 26] + mask[bs, 19, 26]
            out[name] = NestedTensor(x, mask)
        # out['0'] = NestedTensor: tensors[bs, 2048, 19, 26] + mask[bs, 19, 26]
        return out

from .resnet import resnet50
from .resnet import resnet50_se
from .resnet import resnet50_da
from .resnet import resnet50_sa
from .resnet import resnet50_da1
from .resnet import resnet50_ta
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 直接掉包 调用torchvision.models中的backbone
        # 添加我们需要的模块：se、danet、cbnet
        # backbone = resnet50(
        #     pretrain_path='/home/zcs/code/deep-learning-for-image-processing-master/pytorch_object_detection/detr-main/models/resnet50.pth')  # 创建一个resnet50的实例
        backbone = resnet50_ta(
            pretrain_path='/home/zcs/code/deep-learning-for-image-processing-master/pytorch_object_detection/detr-main/models/resnet50.pth')  # 创建一个resnet50的实例
        # backbone = resnet50_se(
        #     pretrain_path='/home/zcs/code/deep-learning-for-image-processing-master/pytorch_object_detection/detr-main/models/resnet50.pth')  # 创建一个resnet50se的实例
        # backbone = resnet50_da(
        #     pretrain_path='./models/resnet50.pth')  # 创建一个resnet50da的实例
        # backbone = resnet50_da1(
        #     pretrain_path='./models/resnet50.pth')  # 创建一个resnet50da的实例
        # backbone = resnet50_sa(
        #     pretrain_path='/home/zcs/code/deep-learning-for-image-processing-master/pytorch_object_detection/detr-main/models/resnet50.pth')  # 创建一个resnet50sa的实例
        backbone.replace_stride_with_dilation = [False, False, dilation]  # 设置replace_stride_with_dilation参数
        backbone.pretrained = is_main_process()  # 设置pretrained参数
        backbone.norm_layer = FrozenBatchNorm2d  # 设置norm_layer参数
        # resnet50  2048
        # backbone = getattr(resnet50(pretrain_path='./backbone/resnet50.pth'))(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
