# import math
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                dilation=dilation, padding=dilation, bias=False)
#         self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(num_groups=2, num_channels=planes* 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.dilation = dilation
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         blocks = [1, 2, 4]
#         if output_stride == 16:
#             strides = [1, 2, 2, 1]
#             dilations = [1, 1, 1, 2]
#         elif output_stride == 8:
#             strides = [1, 2, 1, 1]
#             dilations = [1, 1, 2, 4]
#         else:
#             raise NotImplementedError
#
#         # Modules
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                 bias=False)
#         self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
#         self._init_weight()
#
#         if pretrained:
#             self._load_pretrained_model()
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(num_groups=2, num_channels=planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
#
#         return nn.Sequential(*layers)
#
#     def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(num_groups=2, num_channels=planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
#                             downsample=downsample, BatchNorm=BatchNorm))
#         self.inplanes = planes * block.expansion
#         for i in range(1, len(blocks)):
#             layers.append(block(self.inplanes, planes, stride=1,
#                                 dilation=blocks[i]*dilation, BatchNorm=BatchNorm))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         low_level_feat = x
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x, low_level_feat
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _load_pretrained_model(self):
#         pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
#         model_dict = {}
#         state_dict = self.state_dict()
#         for k, v in pretrain_dict.items():
#             if k in state_dict:
#                 model_dict[k] = v
#         state_dict.update(model_dict)
#         self.load_state_dict(state_dict)
#
# def ResNet101(output_stride, BatchNorm, pretrained=True):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
#     return model
#
# if __name__ == "__main__":
#     import torch
#     model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
#     input = torch.rand(1, 3, 512, 512)
#     output, low_level_feat = model(input)
#     print(output.size())
#     print(low_level_feat.size())

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F
import torch

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=2, num_channels=planes* 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)


    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        out = self.depthwise_conv(x)
        out = out.permute(0, 2, 3, 1)  #
        out = self.norm(out)
        out = F.gelu(out)
        out = self.pointwise_conv1(out)
        out = F.gelu(out)
        out = self.pointwise_conv2(out)
        out = out.permute(0, 3, 1, 2)
        return x + out
# class ConvNeXtBlock(nn.Module):
#     def __init__(self, dim, kernel_size=3, padding=1):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#         self.padding = padding
#
#         # Define multiple branches
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),
#             nn.GroupNorm(num_groups=2, num_channels=dim),
#             nn.ReLU(inplace=True)
#         )
#         # self.branch1 = nn.Conv2d(dim, dim, kernel_size=1, padding=padding, bias=False)
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(num_groups=2, num_channels=dim),
#             nn.ReLU(inplace=True)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False),
#             nn.GroupNorm(num_groups=2, num_channels=dim),
#             nn.ReLU(inplace=True)
#         )
#
#         # Define pointwise convolutions
#         self.pointwise_conv1 = nn.Conv2d(3 * dim, 4 * dim, kernel_size=1, bias=False)
#         self.pointwise_conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         # Apply branches
#         x = x
#         branch1_output = self.branch1(x)
#         branch2_output = self.branch2(x)
#         branch3_output = self.branch3(x)
#
#         # Concatenate branch outputs
#         concatenated_output = torch.cat([branch1_output, branch2_output, branch3_output], dim=1)
#
#         # Apply pointwise convolutions
#         out = self.pointwise_conv1(concatenated_output)
#         out = F.gelu(out)
#         out = self.pointwise_conv2(out)
#
#         # Add residual connection
#         out = x + out
#         return out
class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.enhance_low_level_features_module = nn.Sequential(
            DepthwiseSeparableConv(64, 128, 3, 1, 1),
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0),
            BatchNorm(64),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.depthwise_separable_conv = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1)
        self.convnext_block = ConvNeXtBlock(dim=64)
        self.relu_dsc = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=2, num_channels=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=2, num_channels=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def enhance_low_level_features(self, x):
        return self.enhance_low_level_features_module(x)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        feature_x = x
        x = self.enhance_low_level_features(x)
        x = feature_x + x
        x = self.maxpool(x)


        x = self.convnext_block(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

if __name__ == "__main__":
    import torch
    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())