import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
o_path = os.getcwd()
sys.path.append(os.path.join(o_path, "net"))
from resblock import Bottleneck


class AdaMatting(nn.Module):

    def __init__(self, in_channel):
        self.inplanes = 64
        super(AdaMatting, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder_resblock1 = self._make_layer(Bottleneck, 64, blocks=2, stride=2)
        self.encoder_resblock2 = self._make_layer(Bottleneck, 64, blocks=2, stride=2)
        self.encoder_resblock3 = self._make_layer(Bottleneck, 256, blocks=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # T-decoder
        self.t_decoder_upscale1 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 1024, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale2 = nn.Sequential(
            nn.Conv2d(64 * Bottleneck.expansion, 256, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale3 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 3 * (2 ** 2), kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        # A-deocder
        
        # Propagation unit


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder_conv(x)
        encoder_shallow = self.encoder_maxpool(x)
        encoder_middle = self.encoder_resblock1(encoder_shallow)
        encoder_deep = self.encoder_resblock2(encoder_middle)
        encoder_result = self.encoder_resblock3(encoder_deep)
        t_decoder_deep = self.t_decoder_upscale1(encoder_result) + encoder_deep
        t_decoder_middle = self.t_decoder_upscale2(t_decoder_deep) + encoder_middle
        # t_decoder_shallow = self.t_decoder_upscale3(t_decoder_middle) + encoder_shallow
        return x
