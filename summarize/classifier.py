import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.utils.model_zoo as model_zoo

'''
ResNet, Hanyu 
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConvBlock(nn.Module):
  def __init__(self, in_channel, kernel_size, filters, stride):
    super().__init__()
    f1, f2, f3 = filters
    self.layer = nn.Sequential(
        nn.Conv2d(in_channel, f1, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(f1),
        nn.ReLU(),
        nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=True, bias=False),
        nn.BatchNorm2d(f2),
        nn.ReLU(),
        nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(f3),
        nn.ReLU()
    )
    self.shortcut = nn.Conv2d(in_channel, f3, kernel_size=1, stride=stride, padding=0, bias=False)
    self.batchnorm2d = nn.BatchNorm2d(f3)
    self.relu = nn.ReLU()
  def forward(self, x):
    out = self.layer(x)
    out += self.batchnorm2d(self.shortcut(x))
    out = self.relu(out)
    return out


class IndentityBlock(nn.Module):
  def __init__(self, in_channel, kernel_size, filters):
    super(IndentityBlock,self).__init__()
    f1, f2, f3 = filters
    self.layer = nn.Sequential(
        nn.Conv2d(in_channel, f1 , kernel_size=1,stride=1, padding=0, bias=False),
        nn.BatchNorm2d(f1),
        nn.ReLU(),
        nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=True, bias=False),
        nn.BatchNorm2d(f2),
        nn.ReLU(),
        nn.Conv2d(f2, f3 ,kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(f3),
    )
    self.relu = nn.ReLU(True)
    self.shorcut = nn.Sequential()
  def forward(self, x):
    out = self.layer(x)
    out += self.shorcut(x)
    out = self.relu(out)
    del x
    return out


class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # 3*32*32->64*32*32
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1) # 64**32*32->64*32*32
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2) # 128*16*16-->128*16*16-
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2) # 256*8*8->256*8*8
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2) # 512*4*4->512*4*4
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channels, stride))
            self.in_channel = out_channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    




'''
DenseNet
'''
# planes == channels

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


# block -> Bottleneck
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)



'''
resnet, cong 
'''

__all__ = ['CifarResNet', 'cifar_resnet20', 'cifar_resnet32', 'cifar_resnet44', 'cifar_resnet56']

pretrained_settings = {
    "cifar10": {
        'resnet20': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet20-30abc31d.pth',
        'resnet32': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet32-e96f90cf.pth',
        'resnet44': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet44-f2c66da5.pth',
        'resnet56': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet56-f5939a66.pth',
        'num_classes': 10
    },
    "cifar100": {
        'resnet20': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth',
        'resnet32': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet32-6568a0a0.pth',
        'resnet44': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet44-20aaa8cf.pth',
        'resnet56': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet56-2f147f26.pth',
        'num_classes': 100
    }

}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_resnet20(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [3, 3, 3], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [3, 3, 3], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet20']))
    return model


def cifar_resnet32(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [5, 5, 5], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [5, 5, 5], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet32']))
    return model


def cifar_resnet44(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [7, 7, 7], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [7, 7, 7], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet44']))
    return model


def cifar_resnet56(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [9, 9, 9], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet56']))
    return model



'''
ResNet50 jingxuan
'''

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
# kernel_size=1, padding=0
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)

num_classes = 10
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

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
        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
        
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

'''
VGG16 Bifei
'''
# Reference: https://github.com/kuangliu/pytorch-cifar
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
