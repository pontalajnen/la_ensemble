from torch_uncertainty.layers import PackedConv2d, PackedLinear
from einops import rearrange
from models.frn import PackedFRN, PackedTLU
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock_packed(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_estimators=4, alpha=2, gamma=1):
        super(BasicBlock_packed, self).__init__()
        self.conv1 = PackedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.frn1 = PackedFRN(planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.tlu1 = PackedTLU(planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.conv2 = PackedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.frn2 = PackedFRN(planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.tlu2 = PackedTLU(planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                             num_estimators=num_estimators, alpha=alpha, gamma=gamma),
                PackedFRN(self.expansion * planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
            )

    def forward(self, x):
        out = self.tlu1(self.frn1(self.conv1(x)))
        out = self.frn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.tlu2(out)
        return out


class ResNet_packed(nn.Module):
    def __init__(self, block, num_blocks, num_estimators=4, alpha=2, gamma=1, num_classes=10):
        super(ResNet_packed, self).__init__()
        self.in_planes = 16
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        self.conv1 = PackedConv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha,
                                  gamma=gamma, first=True)
        self.frn1 = PackedFRN(self.in_planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.tlu1 = PackedTLU(self.in_planes, num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = PackedLinear(64 * block.expansion, num_classes, num_estimators=num_estimators,
                                   alpha=alpha, gamma=gamma, last=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                num_estimators=self.num_estimators,
                                alpha=self.alpha,
                                gamma=self.gamma))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tlu1(self.frn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.training:
            return out
        out = rearrange(out, "(m b) c -> b m c", m=self.num_estimators)
        out = out.mean(dim=1)
        return out


def ResNet20_FRN_packed(num_classes):
    return ResNet_packed(BasicBlock_packed, [3, 3, 3], num_classes=num_classes)
