# code taken from https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/models/resnet.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from einops import rearrange
from torch_uncertainty.layers import PackedConv2d, PackedLinear


class FilterResponseNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))

        return torch.maximum(self.gamma * x + self.beta, self.tau)


class PackedFilterResponseNorm2d(nn.Module):
    def __init__(self, num_features, num_estimators=4, alpha=2, gamma=1):
        super().__init__()
        # TODO: Incorporate gamma
        assert num_features % num_estimators == 0, "num_features not divisible by num_estimators"
        self.num_features = int(num_features / num_estimators) * alpha
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        self.filter_response_norms = nn.ModuleList([
            FilterResponseNorm2d(self.num_features)
            for _ in range(num_estimators)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.view(batch_size, self.num_estimators, self.num_features, height, width)

        outputs = []
        for i in range(self.num_estimators):
            out = self.filter_response_norms[i](x[:, i])
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = output.view(batch_size, channels, height, width)

        return output


class PackedBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_estimators=4, alpha=2, gamma=1):
        super().__init__()
        # TODO: Incorporate gamma
        assert num_features % num_estimators == 0, "num_features not divisible by num_estimators"
        self.num_features = int(num_features / num_estimators) * alpha
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(self.num_features)
            for _ in range(num_estimators)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.view(batch_size, self.num_estimators, self.num_features, height, width)

        outputs = []
        for i in range(self.num_estimators):
            out = self.batch_norms[i](x[:, i])
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        output = output.view(batch_size, channels, height, width)

        return output


class BasicBlock_packed(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_estimators=4, alpha=2, gamma=1):
        super(BasicBlock_packed, self).__init__()
        self.conv1 = PackedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.bn1 = PackedBatchNorm2d(planes)
        self.conv2 = PackedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.bn2 = PackedBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                             num_estimators=num_estimators, alpha=alpha, gamma=gamma),
                PackedBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_packed(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, num_estimators=4, alpha=2, gamma=1):
        super(Bottleneck_packed, self).__init__()
        self.conv1 = PackedConv2d(in_planes, planes, kernel_size=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.bn1 = PackedBatchNorm2d(planes)
        self.conv2 = PackedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.bn2 = PackedBatchNorm2d(planes)
        self.conv3 = PackedConv2d(planes, self.expansion*planes, kernel_size=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha, gamma=gamma)
        self.bn3 = PackedBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                             num_estimators=num_estimators, alpha=alpha, gamma=gamma),
                PackedBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_packed(nn.Module):
    def __init__(self, block, num_blocks, num_estimators=4, alpha=2, gamma=1, num_classes=10):
        super(ResNet_packed, self).__init__()
        self.in_planes = 64

        self.conv1 = PackedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                                  num_estimators=num_estimators, alpha=alpha,
                                  gamma=gamma, first=True)
        self.bn1 = PackedBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = PackedLinear(512*block.expansion, num_classes, num_estimators=num_estimators,
                                   alpha=alpha, gamma=gamma, last=True)

        self.num_estimators = num_estimators

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.training:
            return out
        out = rearrange(out, "(m b) c -> b m c", m=self.num_estimators)
        out = out.mean(dim=1)
        return out


# ImageNet models
def ResNet18_packed(num_classes, num_estimators=4, alpha=2, gamma=1):
    return ResNet_packed(BasicBlock_packed, [2, 2, 2, 2], num_classes=num_classes)


def ResNet20_packed():
    return ResNet_packed(BasicBlock_packed)


def ResNet34_packed():
    return ResNet_packed(BasicBlock_packed, [3, 4, 6, 3])


def ResNet50_packed():
    return ResNet_packed(Bottleneck_packed, [3, 4, 6, 3])


def ResNet101_packed():
    return ResNet_packed(Bottleneck_packed, [3, 4, 23, 3])


def ResNet152_packed():
    return ResNet_packed(Bottleneck_packed, [3, 8, 36, 3])


def torch_resnet18(num_classes):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def torch_resnet56(num_classes):
    model = torchvision.models.resnet56(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    # model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model
