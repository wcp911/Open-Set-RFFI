import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

import torch

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return MyConv1dPadSame(in_planes, out_planes, kernel_size=3, stride=stride,groups = 1)           

def conv5x5(in_planes, out_planes, stride=1):
    return MyConv1dPadSame(in_planes, out_planes, kernel_size=5, stride=stride,groups = 1)

def conv7x7(in_planes, out_planes, stride=1):
    return MyConv1dPadSame(in_planes, out_planes, kernel_size=7, stride=stride,groups = 1)

class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
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

class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
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

        d = residual.shape[2] - out.shape[2]
        out1 = residual + out
        out1 = self.relu(out1)
        # out += residual

        return out1

class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
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

        d = residual.shape[2] - out.shape[2]
        out1 = residual + out
        out1 = self.relu(out1)
        # out += residual

        return out1

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, filter_in, filter_out, stride):
        self.filter_in = filter_in
        self.filter_out = filter_out
        self.rate = 2
        super(BasicBlock, self).__init__()

        if filter_in != filter_out:
            downsample_layers = nn.Sequential(
                nn.Conv1d(filter_in, filter_out,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(filter_out),
            )
        else:
            downsample_layers = None

        self.Conv3x3 = BasicBlock3x3(filter_in, filter_out, stride=1, downsample=downsample_layers)
        self.Conv5x5 = BasicBlock5x5(filter_in, filter_out, stride=1, downsample=downsample_layers)
        self.Conv7x7 = BasicBlock7x7(filter_in, filter_out, stride=1, downsample=downsample_layers)

        self.averagepool_1 = torch.nn.AdaptiveAvgPool1d(output_size = 1)

        self.dense = torch.nn.Linear(self.filter_out, int(self.filter_out//self.rate), bias = False)
        self.relu_1 = torch.nn.ReLU()
        self.dense_3x3 = torch.nn.Linear(int(self.filter_out//self.rate), self.filter_out, bias = False)
        self.dense_5x5 = torch.nn.Linear(int(self.filter_out//self.rate), self.filter_out, bias = False)
        self.dense_7x7 = torch.nn.Linear(int(self.filter_out//self.rate), self.filter_out, bias = False)

        self.softmax_1 = torch.nn.Softmax(dim = -1)
    def forward(self, x):
        out_conv3x3 = self.Conv3x3(x)
        out_conv5x5 = self.Conv5x5(x)
        out_conv7x7 = self.Conv7x7(x)

        x_fused_1 = torch.add(out_conv3x3, out_conv5x5, alpha=1)
        x_fused = torch.add(x_fused_1, out_conv7x7, alpha=1)

        x = self.averagepool_1(x_fused) 

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dense(x)
        x = self.relu_1(x)

        x_3x3 = self.dense_3x3(x)
        x_5x5 = self.dense_5x5(x)
        x_7x7 = self.dense_7x7(x)

        x_3x3 = torch.unsqueeze(x_3x3,-1)
        x_5x5 = torch.unsqueeze(x_5x5,-1)
        x_7x7 = torch.unsqueeze(x_7x7,-1)

        x_357 = torch.cat((x_3x3,x_5x5,x_7x7), -1)

        x_357 = self.softmax_1(x_357)

        coff_3 = x_357[:,:,0]
        coff_3 = torch.unsqueeze(coff_3,-1)       
        coff_5 = x_357[:,:,1]
        coff_5 = torch.unsqueeze(coff_5,-1)
        coff_7 = x_357[:,:,2]
        coff_7 = torch.unsqueeze(coff_7,-1)

        out_conv3x3_new = out_conv3x3*coff_3
        out_conv5x5_new = out_conv5x5*coff_5
        out_conv7x7_new = out_conv7x7*coff_7

        x1 = torch.add(out_conv3x3_new,out_conv5x5_new,alpha=1)
        x = torch.add(x1,out_conv7x7_new,alpha=1)

        return x


class SKResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10, feature_dim = 128):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(SKResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.BasicBlock_1 = BasicBlock(64, 64, stride = 1)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.batchnorm_1 = torch.nn.BatchNorm1d(64)

        self.BasicBlock_2 = BasicBlock(64, 128, stride = 1)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.batchnorm_2 = torch.nn.BatchNorm1d(128)

        self.BasicBlock_3 = BasicBlock(128, 256, stride = 1)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=7, stride=2, padding=3)
        self.batchnorm_3 = torch.nn.BatchNorm1d(256)

        self.averagepool_1 = torch.nn.AdaptiveAvgPool1d(output_size = 1)

        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, num_classes)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.BasicBlock_1(x)
        x = self.maxpool_1(x)
        x = self.batchnorm_1(x)

        x = self.BasicBlock_2(x)
        x = self.maxpool_2(x)   
        x = self.batchnorm_2(x)

        x = self.BasicBlock_3(x)
        x = self.maxpool_3(x)
        x = self.batchnorm_3(x)

        x = self.averagepool_1(x)
        x = x.squeeze()

        x = self.drop(x)
        # x:batchsize×256
        out = self.fc(x)
        # out = self.softmax(out)
        return out


