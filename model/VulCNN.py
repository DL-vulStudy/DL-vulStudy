import os
import lap
import torch
import numpy
import time
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
# from sklearn.metrics import precision_recall_fscore_support
from model.score import get_MCM_score, sava_data, load_data

class TraditionalDataset(Dataset):
    def __init__(self, data_df, max_len, hidden_size):
        self.texts = data_df['data']
        self.targets = data_df['val']
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.filename = list(data_df['filename'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3,self.max_len,self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                vectors[j][i] = feature[j][i]
        return {
            'filename':self.filename[idx].split(".")[0],
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }

class TextCNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)            # 卷积核尺寸
        self.num_filters = 32                                          # 卷积核数量(channels数)
        classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state

class VULCNN_Classifier():
    def __init__(self, max_len=100, epochs=100, batch_size = 32, learning_rate = 0.001, \
                item_num = 0, hidden_size = 128):
        self.model = TextCNN(hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        self.item_num = item_num

    def preparation(self, **kw):
        self.result_save_path = kw["result_save_path"]
        self.result_save_path = self.result_save_path + "/" if self.result_save_path[-1]!="/" else self.result_save_path
        model_save_path = self.result_save_path + "model/"        
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        self.result_save_path = self.result_save_path + str(self.item_num) + "_epo" + str(self.epochs) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) + ".result"
        self.model_save_path = model_save_path + str(self.item_num) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) +  "_epo"

        # create datasets
        self.train_set = TraditionalDataset(kw["train_df"], self.max_len, self.hidden_size)
        self.valid_set = TraditionalDataset(kw["test_df"], self.max_len, self.hidden_size)

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

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs,_  = self.model( vectors )
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))   # 获取预测
            labels += list(np.array(targets.cpu()))      # 获取标签

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        filename_dict = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for i, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                filename = data["filename"]
                outputs, _ = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                filename_dict += list(filename)
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        true_dict = []
        false_dict = []
        for count_num in range(len(pre)):
            if pre[count_num] == label[count_num]: true_dict.append(filename_dict[count_num])
            else:false_dict.append(filename_dict[count_num])
        score_dict["report_dict"]["true_dict"] = true_dict
        score_dict["report_dict"]["false_dict"] = false_dict
        val_loss = np.mean(losses)
        return val_loss, score_dict

    
    def train(self):
        best_accuracy_0 = 0
        best_accuracy_1 = 0
        best_accuracy = 0
        learning_record_dict = {}
        # train_table = PrettyTable(['typ', 'epo', 'loss', 'precision', 'recall', 'f_score', 'ACC', 'time'])
        # test_table = PrettyTable(['typ', 'epo', 'loss', 'precision', 'recall', 'f_score', 'ACC', 'time'])
        train_table = PrettyTable(['typ', 'epo', 'loss', 'pre_0','precision', 'rec_0', 'recall',  'f_0', 'f_score', 'ROC',"ACC", 'time'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'pre_0','precision', 'rec_0', 'recall',  'f_0', 'f_score', 'ROC',"ACC", 'time'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            start = time.time()
            train_loss, train_score = self.fit()
            end = time.time()
            train_score["time"] = format(end-start, '.3f')
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "report_dict"])
            print(train_table)

            start = time.time()
            val_loss, val_score = self.eval()
            end = time.time()
            val_score["time"] = format(end-start, '.3f')
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "report_dict"])
            print(test_table)
            print("\n")

            if float(val_score["f_score"]) > best_accuracy or float(val_score["precision"]) > best_accuracy_0 \
                or float(val_score["recall"]) > best_accuracy_1:
                torch.save(self.model.state_dict(), self.model_save_path + str(epoch) + ".pt")
                best_accuracy_0 = float(val_score["precision"])
                best_accuracy_1 = float(val_score["recall"])
                best_accuracy = float(val_score["f_score"])
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                    "train_score": train_score, "val_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")
