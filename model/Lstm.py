import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from model.score import sava_data, load_data
from model.VulCNN import VULCNN_Classifier

class TraditionalDataset(Dataset):
  def __init__(self, data_df, max_len, embeddings):
    self.texts = list(data_df['subcode'])
    self.targets = list(data_df['val'])
    self.max_len = max_len
    self.embeddings = embeddings
    self.embedding_size = len(embeddings[0])
    self.filename = list(data_df['filename'])

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(" ".join(self.texts[idx]))
    target = self.targets[idx]

    vectors = np.zeros(shape=(self.max_len, self.embedding_size))
    for i in range(min(len(self.texts[idx]), self.max_len)):
    # for i in range(min(len(text.split()), self.max_len)):
        # vectors[i] = self.embeddings[text.split()[i]]
        vectors[i] = self.embeddings[self.texts[idx][i]]

    return {
      'filename':self.filename[idx].split(".")[0],
      'text': text,
      'vector': vectors,
      'targets': torch.tensor(target, dtype=torch.long)
    }

class TextRNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextRNN, self).__init__()
        hidden_size = hidden_size #隐藏层数量
        classifier_dropout = 0.25
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2 ##类别数
        num_layers = 2 ##双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # [batch size, 12, hidden_size]
        x = self.dropout(x)
        # [batch size, text size, hidden_size]
        output, (hidden, cell) = self.lstm(x.float())
        # output = [batch size, text size, num_directions * hidden_size]
        output = torch.tanh(output)
        output = self.dropout(output)
        hidden_state = output[:, -1, :]
        output = self.fc(output[:, -1, :])  # 句子最后时刻的 hidden state
        # output = [batch size, num_classes]
        return output, hidden_state

class LSTM_Classifier(VULCNN_Classifier):
    def __init__(self, **kw):
        self.model = TextRNN(kw["hidden_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = kw["max_len"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        self.hidden_size = kw["hidden_size"]
        self.word2vec_model = kw["word2vec_model"]
        self.item_num = kw["item_num"]

    def preparation(self, **kw):
        self.result_save_path = kw["result_save_path"]
        self.result_save_path = self.result_save_path + "/" if self.result_save_path[-1]!="/" else self.result_save_path
        model_save_path = self.result_save_path + "model/"        
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        self.result_save_path = self.result_save_path + str(self.item_num) + "_epo" + str(self.epochs) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) + ".result"
        self.model_save_path = model_save_path + str(self.item_num) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) +  "_epo"
        # create datasets
        self.embeddings = load_data(self.word2vec_model)
        self.train_set = TraditionalDataset(kw["train_df"], self.max_len, self.embeddings)
        self.valid_set = TraditionalDataset(kw["test_df"], self.max_len, self.embeddings)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)