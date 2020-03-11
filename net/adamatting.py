import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from resblock import Bottleneck


class AdaMatting(nn.Module):

    def __init__(self):
        self.inplanes = 64

        # Special attributs
        self.input_space = None
        self.input_size = (320, 320, 4)
        self.mean = None
        self.std = None
        super(AdaMatting, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder_resblock1 = self._make_layer(Bottleneck, 64, blocks=2)
        self.encoder_resblock2 = self._make_layer(Bottleneck, 128, blocks=2, stride=2)
        self.encoder_resblock3 = self._make_layer(Bottleneck, 256, blocks=2, stride=2)

        # T-decoder

        # A-deocder

        # Propagation unit

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
