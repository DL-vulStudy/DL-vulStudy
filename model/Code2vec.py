import os, torch
import sklearn
import pickle
import random
import numpy as np

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from prettytable import PrettyTable
from torch.cuda.amp import autocast


from torch.utils.data import DataLoader
from model.VulCNN import VULCNN_Classifier
from model.score import get_MCM_score, sava_data, load_data, get_MCM_score_code2vec
from transformers import RobertaConfig, RobertaTokenizerFast, BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

class Code2vecDataset(Dataset):
  def __init__(self, **kw):
    self.data = kw["data"]
    self.MAX_LENGTH = 200
    self.BATCH_SIZE = 128
    self.word2idx = kw["word2idx"]
    self.path2idx = kw["path2idx"]
    self.target2idx = kw["target2idx"]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    example = self.data[index]
    name, filename, body = self.parse_line(example)
    length = len(body)
    body += [['<pad>', '<pad>', '<pad>']]*(self.MAX_LENGTH - length)
    temp_n = self.target2idx.get(name, self.target2idx['<unk>'])
    temp_l, temp_p, temp_r = zip(*[(self.word2idx.get(l, self.word2idx['<unk>']), self.path2idx.get(p, self.path2idx['<unk>']), self.word2idx.get(r, self.word2idx['<unk>'])) for l, p, r in body])

    return {
      'filename':filename,
      'temp_l': torch.tensor(temp_l, dtype=torch.long),
      'temp_p': torch.tensor(temp_p, dtype=torch.long),
      'temp_r': torch.tensor(temp_r, dtype=torch.long),
      'targets': torch.tensor(temp_n, dtype=torch.long)
    }

  def parse_line(self, line):
    name, fn, *tree = line.split(' ')
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
    if len(tree) >= 200:
        tree = random.sample(tree, 200)
    assert len(tree) <= 200
    return name, fn, tree




class Code2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super().__init__()
        
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.do = nn.Dropout(dropout)
        
    def forward(self, starts, paths, ends):        
        #starts = paths = ends = [batch size, max length]        
        W = self.W.repeat(starts.shape[0], 1, 1)        
        #W = [batch size, embedding dim, embedding dim * 3]        
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)        
        #embedded_* = [batch size, max length, embedding dim]

        c = self.do(torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2))        
        #c = [batch size, max length, embedding dim * 3]        
        c = c.permute(0, 2, 1)        
        #c = [batch size, embedding dim * 3, max length]
        x = torch.tanh(torch.bmm(W, c))        
        #x = [batch size, embedding dim, max length]        
        x = x.permute(0, 2, 1)        
        #x = [batch size, max length, embedding dim]        
        a = self.a.repeat(starts.shape[0], 1, 1)        
        #a = [batch size, embedding dim, 1]
        z = torch.bmm(x, a).squeeze(2)        
        #z = [batch size, max length]
        z = F.softmax(z, dim=1)        
        #z = [batch size, max length]        
        z = z.unsqueeze(2)        
        #z = [batch size, max length, 1]        
        x = x.permute(0, 2, 1)        
        #x = [batch size, embedding dim, max length]        
        v = torch.bmm(x, z).squeeze(2)        
        #v = [batch size, embedding dim]        
        out = self.out(v)        
        #out = [batch size, output dim]
        return out

class Code2vec_Classifier(VULCNN_Classifier):
    def __init__(self, **kw):
        with open(f"./data/pkl/code2vec/sub_original_dataset/" + kw["file"] + "/" + kw["file"] + ".dict.c2v", 'rb') as file:
            word2count = pickle.load(file)
            path2count = pickle.load(file)
            target2count = pickle.load(file)
            n_training_examples = pickle.load(file)
        word2idx = {'<unk>': 0, '<pad>': 1}
        path2idx = {'<unk>': 0, '<pad>': 1 }
        target2idx = {'<unk>': 0, '<pad>': 1}
        idx2word, idx2path, idx2target = {}, {}, {}
        for w in word2count.keys():          word2idx[w] = len(word2idx)    
        for k, v in word2idx.items():        idx2word[v] = k        
        for p in path2count.keys():          path2idx[p] = len(path2idx)        
        for k, v in path2idx.items():        idx2path[v] = k        
        for t in target2count.keys():        target2idx[t] = len(target2idx)        
        for k, v in target2idx.items():      idx2target[v] = k

        self.word2idx = word2idx
        self.path2idx = path2idx
        self.target2idx = target2idx
        self.model = Code2Vec(len(word2idx), len(path2idx), 128, len(target2idx), 0.25)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_num = kw["item_num"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        # self.hidden_size = 768


    def preparation(self, **kw):
        self.train_df = kw["train_df"]
        self.test_df = kw["test_df"]
        self.result_save_path = kw["result_save_path"]
        self.result_save_path = self.result_save_path + "/" if self.result_save_path[-1]!="/" else self.result_save_path
        model_save_path = self.result_save_path + "model/"        
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        self.result_save_path = self.result_save_path + str(self.item_num) + "_epo" + str(self.epochs) + "_bat" + str(self.batch_size) + "_lr" + str(self.learning_rate) + ".result"
        self.model_save_path = model_save_path + str(self.item_num) + "_bat" + str(self.batch_size) + "_lr" + str(self.learning_rate) +  "_epo"
        # create datasets
        self.train_set = Code2vecDataset(data = self.train_df, word2idx = self.word2idx, path2idx = self.path2idx, target2idx = self.target2idx)
        self.test_set = Code2vecDataset(data = self.test_df, word2idx = self.word2idx, path2idx = self.path2idx, target2idx = self.target2idx)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

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
            tensor_l = data["temp_r"].to(self.device)
            tensor_p = data["temp_r"].to(self.device)
            tensor_r = data["temp_r"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(tensor_l, tensor_p, tensor_r)
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
        score_dict = get_MCM_score_code2vec(labels, predictions)
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
            for _, data in progress_bar:
                tensor_l = data["temp_r"].to(self.device)
                tensor_p = data["temp_r"].to(self.device)
                tensor_r = data["temp_r"].to(self.device)
                targets = data["targets"].to(self.device)
                filename = data["filename"]
                outputs = self.model(tensor_l, tensor_p, tensor_r)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                filename_dict += list(filename)
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.test_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score_code2vec(label, pre)
        true_dict = []
        false_dict = []
        for count_num in range(len(pre)):
            if pre[count_num] == label[count_num]: true_dict.append(filename_dict[count_num])
            else:false_dict.append(filename_dict[count_num])
        score_dict["report_dict"]["true_dict"] = true_dict
        score_dict["report_dict"]["false_dict"] = false_dict
        val_loss = np.mean(losses)
        return val_loss, score_dict
    