import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.backbone.conv2 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.backbone.conv3 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.backbone.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.eval()
    input = torch.rand(1, 3, 256, 512)
    output = model(input)
    print(output.size())
#
#
# import sys
# sys.path.append('./')
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modeling.sync_batchnorm import SynchronizedBatchNorm2d
# from modeling.aspp import ASPP, build_aspp
# from modeling.decoder import Decoder
# from modeling.backbone import build_backbone
#
#
# class AttentionBlock(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionBlock, self).__init__()
#
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#
#         psi = self.psi(psi)
#         return x * psi
#
#
# class AttnDecoder(nn.Module):
#     def __init__(self, num_classes, backbone, BatchNorm):
#         super(AttnDecoder, self).__init__()
#
#         if backbone == 'resnet' or backbone == 'drn':
#             low_level_inplanes = 256
#         elif backbone == 'xception':
#             low_level_inplanes = 128
#         elif backbone == 'mobilenet':
#             low_level_inplanes = 24
#         else:
#             raise NotImplementedError
#
#         self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
#         self.bn1 = BatchNorm(48)
#
#         self.relu = nn.ReLU()
#
#         self.attn = AttentionBlock(48, 48, 48)
#
#         self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm(256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.5),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm(256),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.1),
#                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
#
#     def forward(self, x, low_level_feat):
#
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.bn1(low_level_feat)
#         low_level_feat = self.relu(low_level_feat)
#
#         low_level_feat = self.attn(low_level_feat, low_level_feat)
#
#         x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
#
#         x = torch.cat((x, low_level_feat), dim=1)
#         x = self.last_conv(x)
#
#         return x
#
#
# class DeepLab(nn.Module):
#     def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
#                  sync_bn=True, freeze_bn=False):
#         super(DeepLab, self).__init__()
#
#         if backbone == 'drn':
#             output_stride = 8
#
#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d
#
#         self.backbone = build_backbone(backbone, output_stride, BatchNorm)
#         self.aspp = build_aspp(backbone, output_stride, BatchNorm)
#         self.decoder = AttnDecoder(num_classes, backbone, BatchNorm)
#
#         self.freeze_bn = freeze_bn
#
#     def forward(self, input):
#             x, low_level_feat = self.backbone(input)
#             x = self.aspp(x)
#             x = self.decoder(x, low_level_feat)
#             x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
#
#             return x
#
#     def freeze_bn(self):
#             for m in self.modules():
#                 if isinstance(m, SynchronizedBatchNorm2d):
#                     m.eval()
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.eval()
#
#     def get_1x_lr_params(self):
#             modules = [self.backbone]
#             for i in range(len(modules)):
#                 for m in modules[i].named_modules():
#                     if self.freeze_bn:
#                         if isinstance(m[1], nn.Conv2d):
#                             for p in m[1].parameters():
#                                 if p.requires_grad:
#                                     yield p
#                     else:
#                         if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                                 or isinstance(m[1], nn.BatchNorm2d):
#                             for p in m[1].parameters():
#                                 if p.requires_grad:
#                                     yield p
#
#     def get_10x_lr_params(self):
#             modules = [self.aspp, self.decoder]
#             for i in range(len(modules)):
#                 for m in modules[i].named_modules():
#                     if self.freeze_bn:
#                         if isinstance(m[1], nn.Conv2d):
#                             for p in m[1].parameters():
#                                 if p.requires_grad:
#                                     yield p
#                     else:
#                         if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                                 or isinstance(m[1], nn.BatchNorm2d):
#                             for p in m[1].parameters():
#                                 if p.requires_grad:
#                                     yield p
#
# if __name__ == "__main__":
#     model = DeepLab(output_stride=16)
#     model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.backbone.conv2 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.backbone.conv3 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.backbone.con4 = nn.ConvTranspose2d(256, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.eval()
#     input = torch.rand(1, 3, 256, 512)
#     output = model(input)
#     print(output.size())
