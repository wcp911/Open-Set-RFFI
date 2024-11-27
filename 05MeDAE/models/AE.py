import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np

class RSBU_CS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, kernel_size=3):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding=torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result

class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.RSBU_block = nn.Sequential(
            RSBU_CS(in_channels=16, out_channels=16, down_sample=False),
            RSBU_CS(in_channels=16, out_channels=32, down_sample=False),

            RSBU_CS(in_channels=32, out_channels=64, down_sample=True),
            RSBU_CS(in_channels=64, out_channels=128, down_sample=True),
            RSBU_CS(in_channels=128, out_channels=256, down_sample=True),
            RSBU_CS(in_channels=256, out_channels=512, down_sample=True),
            RSBU_CS(in_channels=512, out_channels=1024, down_sample=True),
        )

        self.GAP = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.RSBU_block(x)
        x = self.GAP(x)
        x = torch.squeeze(x, -1)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc = nn.Linear(1024, 1024)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),)
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),)

        self.conv7 = nn.Conv1d(in_channels=64, out_channels=in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, input):
        x = self.fc(input)
        batchsize, len = x.shape
        x = torch.reshape(x, (batchsize, 64, int(len/64)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        batchsize, _, len = x.shape
        x = torch.reshape(x, (batchsize, self.in_channels, len))

        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, num_class):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)
        self.Classifier = nn.Linear(1024, num_class)

    def forward(self, input):
        encoder_out = self.encoder(input)
        classifier_out = self.Classifier(encoder_out)
        decoder_out = self.decoder(encoder_out)

        return encoder_out, classifier_out, decoder_out


if __name__ == '__main__':
    image = torch.randn(4, 2, 1024)
    net = AutoEncoder(in_channels=2, num_class=9)
    encoder_out, classifier_out, decoder_out = net(image)
    print(classifier_out)
