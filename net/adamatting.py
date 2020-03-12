import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
o_path = os.getcwd()
sys.path.append(os.path.join(o_path, "net"))
from resblock import Bottleneck
from gcn import GCN


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
        # Shortcuts
        self.shortcut_shallow = GCN(64, 32)
        self.shortcut_middle = GCN(64 * Bottleneck.expansion, 64)
        self.shortcut_deep = GCN(64 * Bottleneck.expansion, 256)
        # T-decoder
        self.t_decoder_upscale1 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 256 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale2 = nn.Sequential(
            nn.Conv2d(256, 64 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale4 = nn.Sequential(
            nn.Conv2d(32, 3 * (2 ** 2), kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        # A-deocder
        self.a_decoder_upscale1 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 256 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale2 = nn.Sequential(
            nn.Conv2d(256, 64 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale4 = nn.Sequential(
            nn.Conv2d(32, 1 * (2 ** 2), kernel_size=7, stride=1, padding=3, bias=True),
            nn.PixelShuffle(2)
        )
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
        x = self.encoder_conv(x) # 64
        encoder_shallow = self.encoder_maxpool(x) # 64
        encoder_middle = self.encoder_resblock1(encoder_shallow) # 256
        encoder_deep = self.encoder_resblock2(encoder_middle) # 256
        encoder_result = self.encoder_resblock3(encoder_deep) # 1024

        shortcut_deep = self.shortcut_deep(encoder_deep)
        shortcut_middle = self.shortcut_middle(encoder_middle)
        shortcut_shallow = self.shortcut_shallow(encoder_shallow)

        t_decoder_deep = self.t_decoder_upscale1(encoder_result) + shortcut_deep # 256
        t_decoder_middle = self.t_decoder_upscale2(t_decoder_deep) + shortcut_middle # 64
        t_decoder_shallow = self.t_decoder_upscale3(t_decoder_middle) # 32
        t_decoder = self.t_decoder_upscale4(t_decoder_shallow)

        a_decoder_deep = self.a_decoder_upscale1(encoder_result)
        a_decoder_middle = self.a_decoder_upscale2(a_decoder_deep) + shortcut_middle # 64
        a_decoder_shallow = self.a_decoder_upscale3(a_decoder_middle) + shortcut_shallow # 32
        a_decoder = self.a_decoder_upscale4(a_decoder_shallow)


        return t_decoder
