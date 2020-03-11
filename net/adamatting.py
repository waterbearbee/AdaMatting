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
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder_resblock1 = self._make_layer(Bottleneck, 64, blocks=2)
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

    def forward(self, input):
        x = self.encoder_conv(input)
        x = self.encoder_maxpool(x)
        x = self.encoder_resblock1(x)
        x = self.encoder_resblock2(x)
        x = self.encoder_resblock3(x)
        return x
