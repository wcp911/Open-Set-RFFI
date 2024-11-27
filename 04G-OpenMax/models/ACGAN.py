import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, num_class):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        # first linear layer
        self.fc = nn.Linear(100, 1024 * in_channels)

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, 7, 1, 3, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels, in_channels, 7, 1, 3, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels, in_channels, 7, 1, 3, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels, in_channels, 7, 1, 3, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels, in_channels, 7, 1, 3, bias=False),
            nn.Tanh(),
        )
        self.label_emb = nn.Embedding(num_class, 100)

    def forward(self, input, labels):
        input = torch.mul(self.label_emb(labels), input)
        x = self.fc(input)
        B, len = x.shape
        x = torch.reshape(x, (B, self.in_channels, int(len/self.in_channels)))
        x = self.conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_class):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),)
        self.conv9 = nn.Sequential(
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 512)
        # discriminator fc
        self.fc_dis = nn.Linear(512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(512, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.flatten(x)
        x = self.fc(x)
        discriminator = self.fc_dis(x)
        classifier = self.fc_aux(x)
        realfake = self.sigmoid(discriminator).view(-1, 1)
        classes = self.softmax(classifier)

        return realfake, classes


if __name__ == '__main__':
    image = torch.randn(4, 2, 1024)
    net = Discriminator(in_channels=2, num_class=9)
    realfake, classes = net(image)
    print(realfake)

    image = torch.randn(4, 100)
    net = Generator(in_channels=3, num_class=10)
    data = net(image)
    print(data)
