import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36
    """

    def __init__(self, in_channels, out_channels, activation=None):
        super(DenseBlock, self).__init__()
        if activation == 'sig':
            self.activation = nn.Sigmoid()
        if activation == 'softmax':
            self.activation = nn.Softmax()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.bn = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.activation(self.conv1(bn))
        conv2 = self.activation(self.conv2(conv1))

        c2_dense = self.activation(torch.cat([conv1, conv2], 1))
        conv3 = self.activation(self.conv3(c2_dense))
        c3_dense = self.activation(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.activation(self.conv4(c3_dense))
        c4_dense = self.activation(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.activation(self.conv5(c4_dense))
        c5_dense = self.activation(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(40),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            DenseBlock(50, 1, 'relu')
        )
        self.linear = nn.Linear(903, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=7, stride=3, padding=1),
            nn.BatchNorm1d(10),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(20),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(30),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(40),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),

            nn.ConvTranspose1d(40, 30, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.ConvTranspose1d(30, 20, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.ConvTranspose1d(20, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.ConvTranspose1d(10, 5, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(5),
            # -nn.ReLU(),
            nn.ConvTranspose1d(5, 1, kernel_size=3, stride=1, padding=1),
            # -nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)

        return x
