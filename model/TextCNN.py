import torch, os
from torch import nn
import torch.nn.functional as F
from model.Lstm import LSTM_Classifier

class TextCNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNN, self).__init__()
        hidden_size = hidden_size #隐藏层数量
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2 ##类别数
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state

class TextCNN_Classifier(LSTM_Classifier):
    def __init__(self, **kw):
        self.model = TextCNN(kw["hidden_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = kw["max_len"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        self.hidden_size = kw["hidden_size"]
        self.word2vec_model = kw["word2vec_model"]
        self.item_num = kw["item_num"]