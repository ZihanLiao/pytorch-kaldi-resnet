import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
#from scipy import linalg as la
import torch.nn.functional as F
from densenet import densenet62

# Author: Nanxin Chen, Cheng-I Lai

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ThinResNet(nn.Module):
    """ResNet with smaller channel dimensions
    """
    def __init__(self, block, layers):
        self.inplanes = 8
        super(ThinResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((1, 3))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d((1, 3))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        #print(x.shape) # 128, 1, 800, 30
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape) # 128, 128, 100, 4

        #x = self.avgpool(x)
        #print(x.shape) # 128, 128, 100, 1
        #x = x.view(x.size(0), x.size(1), x.size(2)).permute(0, 2, 1)
        #print(x.shape) # 128, 100, 128

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def thin_resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ThinResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class NeuralSpeakerModel(nn.Module):
    """Neural Speaker Model 
    @spk_num: number of speakers
    @distance_tpye: 1) norm (Frobenius Norm) or 2) sqr (square norm) --> distance metric in Eq(4) in LDE paper, for calculating the weight over the residual vectors
    @network_type: 1) att (multi-head attention, or attention over T) or 2) lde (LDE, or attention over dictionary components).
    @pooling: aggregation step over the residual vectors 1) mean only or 2) mean and std
    """
    def __init__(self, spk_num, feat_dim=40, pooling='mean'):
        super(NeuralSpeakerModel, self).__init__()

        self.res = resnet34()

        _feature_dim = (feat_dim+7) // 8 #default 5
        #self.avgpool = nn.AvgPool2d((1, 3))
        #self.fc1 = nn.Linear(256 * block.expansion, 256)
        self.pool = StatsPooling(pooling=pooling)
        self.flat = nn.Flatten(1, -1)
        if pooling=='mean':
            #self.pool = nn.AvgPool2d((1, 25)) #frames/8 
            self.fc1  = nn.Linear(_feature_dim*256, 256)
        if pooling=='mean+std':
            #self.pool = nn.AvgPool2d((1,3)) #how to get mean+std
            self.fc1  = nn.Linear(_feature_dim*2*256, 256)
        self.bn1  = nn.BatchNorm1d(256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(256, spk_num)

    def forward(self, x):
        #print('input size: {}'.format(x.shape))
        x = self.res(x) #128, 256, 5, 25
        #print('layer res size: {}'.format(x.shape))

        x = self.pool(x) #128, 256, 5/10, 1
        #print('layer pool size: {}'.format(x.shape))
        x = self.flat(x)
        #print('layer flat size: {}'.format(x.shape))

        x = self.fc1(x)
        #print('layer fc1 size: {}'.format(x.shape))
        x = self.bn1(x)
        x = self.fc1_relu(x)

        x = self.fc2(x)
        #print('layer fc2 size: {}'.format(x.shape))
        #return F.log_softmax(x, dim=-1)
	# use torch.nn.CrossEntropyLoss()
        return x

    def predict(self, x):
        x = self.res(x)
        x = self.pool(x)
        x = self.flat(x)
        #if type(x) is tuple:
        #    x = x[0]
        x = self.fc1(x)
        return x


class AngleLoss(nn.Module):
    # source: https://github.com/clcarwin/sphereface_pytorch
    # AngleLoss class
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte().detach()
        #index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

class StatsPooling(nn.Module):
    def __init__(self, pooling='mean'):
        super(StatsPooling, self).__init__()
        self.pooling = pooling
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
            
    def forward(self, input):
        if self.pooling == 'mean':
            #return torch.sum(input, dim=1) # Eq(5) in LDE paper
            output = self.pool(input)
            return output
        elif self.pooling == 'mean+std':
            #mean = torch.sum(input, dim=3) # Eq(5) in LDE paper
            #std = torch.sqrt(torch.var(input, dim=3))
            #std = torch.sqrt(torch.sum((x-mean)** 2, dim=3)+1e-8) # std vector
            mean, var = torch.var_mean(input, dim=3)
            #print('layer mean size: {}'.format(mean.shape))
            #print('layer var size: {}'.format(var.shape))
            std = torch.sqrt(var)
            output = torch.cat([mean, std], dim=-1)
            return output
        else:
            raise NotImplementedError
