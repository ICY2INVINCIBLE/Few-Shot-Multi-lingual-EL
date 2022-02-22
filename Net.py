import torch.nn as nn
from collections import OrderedDict
import torch

# class LanguageDetector(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  hidden_size,
#                  dropout,
#                  batch_norm=False):
#         super(LanguageDetector, self).__init__()
#         assert num_layers >= 0, 'Invalid layer numbers'
#         self.net = nn.Sequential()
#         for i in range(num_layers):
#             if dropout > 0:
#                 self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
#             self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
#             if batch_norm:
#                 self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
#             self.net.add_module('q-relu-{}'.format(i), nn.ReLU())
#
#         self.net.add_module('q-linear-final', nn.Linear(hidden_size, 1))
#
#     def forward(self, input):
#         return self.net(input)
#
# class EntityClassifier(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  hidden_size,
#                  output_size,
#                  dropout,
#                  batch_norm=False):
#         super(EntityClassifier, self).__init__()
#         assert num_layers >= 0, 'Invalid layer numbers'
#         self.net = nn.Sequential()
#         for i in range(num_layers):
#             if dropout > 0:
#                 self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
#             self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
#             if batch_norm:
#                 self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
#             self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
#
#         self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
#         self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))
#
#     def forward(self, input):
#         return self.net(input)
#
# class LeNetSequentialOrderDict(nn.Module):
#     def __init__(self, classes):
#         super(LeNetSequentialOrderDict, self).__init__()
#
#         self.features = nn.Sequential(OrderedDict({  # OrderedDict可以实现自命名各个网络层
#             'conv1': nn.Conv2d(3, 6, 5),
#             'relu1': nn.ReLU(inplace=True),
#             'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
#
#             'conv2': nn.Conv2d(6, 16, 5),
#             'relu2': nn.ReLU(inplace=True),
#             'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
#         }))
#
#         self.classifier = nn.Sequential(OrderedDict({
#             'fc1': nn.Linear(16 * 5 * 5, 120),
#             'relu3': nn.ReLU(),
#
#             'fc2': nn.Linear(120, 84),
#             'relu4': nn.ReLU(inplace=True),
#
#             'fc3': nn.Linear(84, classes),
#         }))
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size()[0], -1)
#         x = self.classifier(x)
#         return x


class LangDetectNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_layers: int, hidden_dim: int,drop_out: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            dropout=drop_out,
                            bidirectional=True,
                            batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_dim, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x: torch.TensorType):
        # shape of x: [seq_len, batch_size]
        # input_x=torch.LongTensor(input_x)
        x = self.embedding(input_x)
        # shape of x: [seq_len, batch_size, embedding_dim]
        outp, (hidden, cell) = self.LSTM(x)

        # shape of outp: [seq_len, batch_size, 2*hidden_dim]
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.relu(self.fc1(hidden_last))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x