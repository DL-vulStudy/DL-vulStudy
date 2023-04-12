import os, torch
import sklearn
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from prettytable import PrettyTable
from torch.cuda.amp import autocast

from torch.utils.data import DataLoader
from model.VulCNN import VULCNN_Classifier
from model.score import get_MCM_score, sava_data, load_data
from transformers import RobertaConfig, RobertaTokenizerFast, BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

class RobertaDataset(Dataset):
  def __init__(self, **kw):
    self.data = kw["data"]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    codes = self.data.subcode.iloc[index]
    target = self.data.val.iloc[index]



    return {
      'filename':self.data.filename.iloc[index].split(".")[0],
      'codes': torch.tensor(codes, dtype=torch.long),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class myCNN(nn.Module):
    def __init__(self, EMBED_SIZE, EMBED_DIM):
        super(myCNN,self).__init__()
        
        pretrained_weights = RobertaModel.from_pretrained('/root/data/qm_data/VulBERTa/models/VulBERTa/').embeddings.word_embeddings.weight

        self.embed = nn.Embedding.from_pretrained(pretrained_weights,
                                                  freeze=True,
                                                  padding_idx=1)

        self.conv1 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200*3,256) #500
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,2)
    
    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0,2,1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        x1 = F.max_pool1d(x1, x1.shape[2])
        x2 = F.max_pool1d(x2, x2.shape[2])
        x3 = F.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x1,x2,x3],dim=1)
        
        # flatten the tensor
        x = x.flatten(1)
        
        # apply mean over the last dimension
        #x = torch.mean(x, -1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)

class Roberta_Classifier(VULCNN_Classifier):
    def __init__(self, **kw):
        self.model = myCNN(50002, 768)
        self.model.embed.weight.data[1] = torch.zeros(768)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_num = kw["item_num"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        self.hidden_size = 768


    def preparation(self, **kw):
        self.train_df = kw["train_df"]
        self.test_df = kw["test_df"]
        self.result_save_path = kw["result_save_path"]
        self.result_save_path = self.result_save_path + "/" if self.result_save_path[-1]!="/" else self.result_save_path
        model_save_path = self.result_save_path + "model/"        
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        self.result_save_path = self.result_save_path + str(self.item_num) + "_epo" + str(self.epochs) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) + ".result"
        self.model_save_path = model_save_path + str(self.item_num) + "_bat" + str(self.batch_size) + "_dim" + str(self.hidden_size) + "_lr" + str(self.learning_rate) +  "_epo"
        # create datasets
        self.train_set = RobertaDataset(data = self.train_df)
        self.test_set = RobertaDataset(data = self.test_df)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
        # self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)
        # self.valid_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)

        # helpers initialization
        cw = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=[0,1],y=list(self.train_df.val))
        c_weights = torch.FloatTensor([cw[0], cw[1]])
        self.loss_fn  = nn.CrossEntropyLoss(weight=c_weights).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            codes = data["codes"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs = self.model(codes)
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
            for _, data in progress_bar:
                codes = data["codes"].to(self.device)
                targets = data["targets"].to(self.device)
                filename = data["filename"]
                outputs = self.model(codes)
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
    