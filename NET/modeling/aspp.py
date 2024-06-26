# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#
# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                             stride=1, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.GroupNorm(num_groups=2, num_channels=planes)
#         self.relu = nn.ReLU()
#
#         self._init_weight()
#
#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)
#
#         return self.relu(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
# class ASPP(nn.Module):
#     def __init__(self, backbone, output_stride, BatchNorm):
#         super(ASPP, self).__init__()
#         if backbone == 'drn':
#             inplanes = 512
#         elif backbone == 'mobilenet':
#             inplanes = 320
#         else:
#             inplanes = 2048
#         if output_stride == 16:
#             dilations = [1, 6, 12, 18]
#         elif output_stride == 8:
#             dilations = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError
#
#         self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
#
#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
#                                              nn.GroupNorm(num_groups=2, num_channels=256),
#                                              nn.ReLU())
#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         self.bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()
#
#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         return self.dropout(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2. / n))
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# def build_aspp(backbone, output_stride, BatchNorm):
#     return ASPP(backbone, output_stride, BatchNorm)

# Import necessary modules
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# ASPP Module with additional convolutions
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(num_groups=2, num_channels=planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ASPP with added convolutions at the end
class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        # Initial setup
        inplanes = 2048 if backbone not in ['drn', 'mobilenet'] else 512 if backbone == 'drn' else 320
        dilations = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]

        # ASPP modules
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.GroupNorm(num_groups=2, num_channels=256),
                                             nn.ReLU())

        # Convolutional layers added
        self.extra_conv1 = nn.Conv2d(1280, 256, kernel_size=3, padding=1, bias=False)
        self.extra_bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.extra_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.extra_bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # Passing through the additional convolutions
        x = F.relu(self.extra_bn1(self.extra_conv1(x)))
        x = F.relu(self.extra_bn2(self.extra_conv2(x)))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# Function to build ASPP
def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


# Example usage
if __name__ == "__main__":
    # Setup for demonstration
    model = ASPP(backbone='resnet', output_stride=16, BatchNorm=nn.BatchNorm2d)
    model.eval()
    input = torch.rand(1, 2048, 64, 128)  # Example input tensor
    output = model(input)
    print("Output size:", output.size())
