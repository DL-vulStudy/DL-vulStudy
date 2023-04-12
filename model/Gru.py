import torch, os
from torch import nn
from model.Lstm import LSTM_Classifier

class TextRNN_GRU(nn.Module):
    def __init__(self, hidden_size):
        super(TextRNN_GRU, self).__init__()
        hidden_size = hidden_size #隐藏层数量
        classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2 ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch size, 12, hidden_size]
        x = self.dropout(x)
        # [batch size, text size, hidden_size]
        output, hidden = self.lstm(x.float())
        # output = [batch size, text size, num_directions * hidden_size]
        output = torch.tanh(output)
        output = self.dropout(output)
        hidden_state = output[:, -1, :]
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output, hidden_state

class GRU_Classifier(LSTM_Classifier):
    def __init__(self, **kw):
        self.model = TextRNN_GRU(kw["hidden_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = kw["max_len"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        self.hidden_size = kw["hidden_size"]
        self.word2vec_model = kw["word2vec_model"]
        self.item_num = kw["item_num"]
