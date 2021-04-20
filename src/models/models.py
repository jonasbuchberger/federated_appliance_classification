import torch
import torch.nn as nn


class BlondConvNet(nn.Module):

    def __init__(self, in_features, seq_len, num_classes):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            num_classes(int): Number of targets
        """
        super(BlondConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_features, 10, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv1d(10, 15, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(15),
            nn.ReLU())

        seq_len = self._calc_dims(self.layer1, seq_len)
        seq_len = self._calc_dims(self.layer2, seq_len)

        self.classifier = nn.Sequential(
            nn.Linear(seq_len * 15, num_classes)
        )

    def _calc_dims(self, layer, seq_len):
        """
        Args:
            layer (nn.Sequential): Current layer
            seq_len (int): Length of input series

        Returns:
            (int): Series length of the layer output
        """
        seq_len = seq_len + (2 * layer[0].padding[0]) - layer[0].dilation[0] * (layer[0].kernel_size[0] - 1) - 1 // \
                  layer[0].stride[0] + 1

        return seq_len

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondLstmNet(nn.Module):

    def __init__(self, in_features, seq_len, num_classes, batch_size, hidden_layer_size=15, num_layers=1):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            num_classes(int): Number of targets
            batch_size (int: Number of samples in one batch
            hidden_layer_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
        """
        super(BlondLstmNet, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(in_features, self.hidden_layer_size, dropout=0, num_layers=self.num_layers,
                            batch_first=True)

        self.hidden_cell = (torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size),
                            torch.zeros(self.num_layers, self.batch_size, self.hidden_layer_size))

        self.classifier = nn.Linear(self.hidden_layer_size * self.seq_len, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    in_features = 3
    seq_len = 10
    bs = 7

    model = BlondLstmNet(in_features, seq_len, 5, bs)
    model1 = BlondConvNet(in_features, seq_len, 5)

    x = torch.rand((bs, in_features, seq_len))

    # print(x)
    pred = model1(x)

    from sklearn.metrics import precision_recall_fscore_support
    pred = [1,2,3,4,1,2,3,4]
    target = [1,2,3,5,1,5,3,5]
    _,_,f1,_ = precision_recall_fscore_support(target, pred, average='macro')
    print(f1)
