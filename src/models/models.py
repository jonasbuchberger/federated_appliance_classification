import torch
import torch.nn as nn


class BlondConvNet(nn.Module):

    def __init__(self, in_features, seq_len, num_classes, out_features=10, num_layers=1):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            num_classes (int): Number of targets
            out_features (int): Size of first out_channels of convolutional block
            num_layers (int): Number of stacked layer blocks
        """
        super(BlondConvNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            layer = BlondConvNetLayer(in_features, seq_len, out_features)
            self.layers.append(layer)

            # Assign values for next layer
            seq_len = layer.seq_len
            in_features = out_features
            out_features = int(out_features * 1.5)

        self.classifier = BlondNetMLP(seq_len, in_features, num_classes, max(1, int(num_layers / 2)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondConvNetLayer(nn.Module):

    def __init__(self, in_features, seq_len, out_features):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            out_features (int): Size of hidden layer
        """
        super(BlondConvNetLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.ReLU())

        self.seq_len = self._calc_dims(self.layer, seq_len)

    def _calc_dims(self, layer, seq_len):
        """
        Args:
            layer (nn.Sequential): Current layer
            seq_len (int): Length of input series

        Returns:
            (int): Series length of the layer output
        """
        seq_len = int((seq_len + (2 * layer[0].padding[0]) - layer[0].dilation[0] * (layer[0].kernel_size[0] - 1) - 1) /
                      layer[0].stride[0] + 1)

        seq_len = int((seq_len + (2 * layer[2].padding) - layer[2].dilation * (layer[2].kernel_size - 1) - 1) / layer[
            2].stride + 1)

        return seq_len

    def forward(self, x):
        x = self.layer(x)

        return x


class BlondNetMLP(nn.Module):

    def __init__(self, seq_len, in_features, num_classes, num_layers):
        super(BlondNetMLP, self).__init__()

        in_size = in_features * seq_len
        self.mlp = nn.Sequential()
        i = 0
        for i in range(1, num_layers):
            self.mlp.add_module(f'Linear({i - 1})', nn.Linear(in_size, int(in_size / 2)))
            in_size = int(in_size / 2)

        self.mlp.add_module(f'Linear({i})', nn.Linear(in_size, num_classes))

    def forward(self, x):
        x = self.mlp(x)

        return x


class BlondLstmNet(nn.Module):

    def __init__(self, in_features, seq_len, num_classes, hidden_layer_size=15, num_layers=1):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            num_classes(int): Number of targets
            hidden_layer_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
        """
        super(BlondLstmNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(in_features, hidden_layer_size, dropout=0, num_layers=num_layers,
                            batch_first=True)

        self.classifier = BlondNetMLP(seq_len, hidden_layer_size, num_classes, max(1, int(num_layers / 2)))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).requires_grad_()

        x = x.transpose(2, 1)
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondResNet(nn.Module):

    def __init__(self, in_features, num_classes, out_features=10, num_layers=1):
        """
        Args:
            in_features (int): Number of input features
            num_classes (int): Number of targets
            out_features (int): Size of first out_channels of convolutional block
            num_layers (int): Number of stacked layer blocks
        """
        super(BlondResNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            layer = BlondResNetLayer(in_features, out_features)
            self.layers.append(layer)

            # Assign values for next layer
            in_features = out_features
            out_features = int(out_features * 1.5)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = BlondNetMLP(1, in_features, num_classes, max(1, int(num_layers / 2)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondResNetLayer(nn.Module):

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): Number of input features
            out_features (int): Size of hidden layer
        """
        super(BlondResNetLayer, self).__init__()

        self.skip_connection = None
        if in_features != out_features:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_features))

        self.layer = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features))

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.layer(x)
        if self.skip_connection:
            identity = self.skip_connection(identity)
        x += identity
        x = self.relu(x)

        return x


class BlondDenseNet(nn.Module):

    def __init__(self, in_features, num_classes, out_features=10, num_layers=1):
        """
        Args:
            in_features (int): Number of input features
            out_features (int): Size of first out_channels of convolutional block
            num_layers (int): Number of stacked layer blocks
        """
        super(BlondDenseNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            layer = BlondDenseLayer(in_features, out_features)
            self.layers.append(layer)

            if i < num_layers - 1:
                transition = BlondDenseTransitionLayer(in_features + 2 * out_features, 64)
                self.layers.append(transition)

                in_features = 64
                out_features = int(out_features * 1.5)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = BlondNetMLP(1, in_features + 2 * out_features, num_classes, max(1, int(num_layers / 2)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondDenseLayer(nn.Module):
    """
    Args:
        in_features (int): Number of input features
        out_features (int): Size of out_channels of convolutional block
    """

    def __init__(self, in_features, out_features):
        super(BlondDenseLayer, self).__init__()

        self.layer1 = nn.Sequential(nn.BatchNorm1d(in_features),
                                    nn.ReLU(),
                                    nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm1d(out_features),
                                    nn.ReLU(),
                                    nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1))

        in_features = in_features + out_features
        self.layer2 = nn.Sequential(nn.BatchNorm1d(in_features),
                                    nn.ReLU(),
                                    nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm1d(out_features),
                                    nn.ReLU(),
                                    nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        lay1 = self.layer1(x)
        x1 = torch.cat([x, lay1], 1)
        lay2 = self.layer2(x1)
        out = torch.cat([x, lay1, lay2], 1)
        return out


class BlondDenseTransitionLayer(nn.Module):
    """
   Args:
       in_features (int): Number of input features
       out_features (int): Size of out_channels of convolutional block
   """
    def __init__(self, in_features, out_features):

        super(BlondDenseTransitionLayer, self).__init__()

        self.transition = nn.Sequential(nn.BatchNorm1d(in_features),
                                        nn.ReLU(),
                                        nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False),
                                        nn.AvgPool1d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.transition(x)
        return x


if __name__ == '__main__':
    in_features = 64
    seq_len = 190
    bs = 7

    model = BlondLstmNet(in_features, seq_len, 5)
    model1 = BlondConvNet(in_features, seq_len, 5)
    model2 = BlondResNet(in_features, 5, num_layers=4, out_features=28)
    model3 = BlondDenseNet(in_features, 5, num_layers=4, out_features=32)

    x = torch.rand((bs, in_features, seq_len))

    # print(x)
    pred = model1(x)
    print(pred.shape)
    pred = model(x)
    print(pred.shape)
    pred = model2(x)
    print(pred.shape)
    pred = model3(x)
    print(pred.shape)

