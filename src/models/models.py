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
            nn.Conv1d(in_features, out_features, kernel_size=2, stride=1, padding=1),
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


if __name__ == '__main__':
    in_features = 3
    seq_len = 25
    bs = 7

    model = BlondLstmNet(in_features, seq_len, 5, bs)
    model1 = BlondConvNet(in_features, seq_len, 5)

    x = torch.rand((bs, in_features, seq_len))

    # print(x)
    pred = model1(x)
    print(pred.shape)
    pred = model(x)
    print(pred.shape)

    """
    from sklearn.metrics import precision_recall_fscore_support

    pred = [1, 2, 3, 4, 1, 2, 3, 4]
    target = [1, 2, 3, 5, 1, 5, 3, 5]
    _, _, f1, _ = precision_recall_fscore_support(target, pred, average='macro')
    print(f1)
    """
