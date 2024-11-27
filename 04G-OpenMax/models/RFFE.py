import torch
import torch.nn as nn

class RFFE(nn.Module):
    def __init__(self, in_channels, num_class):
        super(RFFE, self).__init__()

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
        self.fc = nn.Linear(1024, num_class)

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

        return x


if __name__ == '__main__':
    image = torch.randn(4, 2, 1024)
    net = RFFE(in_channels=2, num_class=9)
    classifier = net(image)
    print(classifier)


