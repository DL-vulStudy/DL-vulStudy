import numpy as np
import torch, os
from tqdm import tqdm
from torch import nn
from torch.cuda.amp import autocast
from model.score import get_MCM_score
from model.devign import Devign_Classifier
from transformers import AdamW, get_linear_schedule_with_warmup

class MetricLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.2, aplha=0.5, lambda1=0.5, lambda2=0.001, num_layers=1):
        super(MetricLearningModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.dropout_p = dropout_p
        self.alpha = aplha
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            nn.LogSoftmax(dim=-1)
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_function = nn.NLLLoss(reduction='none')
        # print(self.alpha, self.lambda1, self.lambda2, sep='\t', end='\t')

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(self, example_batch,
                targets=None,
                positive_batch=None,
                negative_batch=None):
        train_mode = (positive_batch is not None and
                      negative_batch is not None and
                      targets is not None)
        h_a = self.extract_feature(example_batch)
        y_a = self.classifier(h_a)
        probs = torch.exp(y_a)
        batch_loss = None
        if targets is not None:
            targets=torch.reshape(targets,(targets.size()[0],1))
            ce_loss = self.loss_function(input=y_a, target=targets.squeeze())
            batch_loss = ce_loss.sum(dim=-1)
        if train_mode:
            h_p = self.extract_feature(positive_batch)
            h_n = self.extract_feature(negative_batch)
            dot_p = h_a.unsqueeze(dim=1) \
                .bmm(h_p.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            dot_n = h_a.unsqueeze(dim=1) \
                .bmm(h_n.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            mag_a = torch.norm(h_a, dim=-1)
            mag_p = torch.norm(h_p, dim=-1)
            mag_n = torch.norm(h_n, dim=-1)
            D_plus = 1 - (dot_p / (mag_a * mag_p))
            D_minus = 1 - (dot_n / (mag_a * mag_n))
            trip_loss = self.lambda1 * torch.abs((D_plus - D_minus + self.alpha))
            ce_loss = self.loss_function(input=y_a, target=targets.squeeze())
            l2_loss = self.lambda2 * (mag_a + mag_p + mag_n)
            total_loss = ce_loss + trip_loss + l2_loss
            batch_loss = (total_loss).sum(dim=-1)
        return probs, h_a, batch_loss




class Reveal_Classifier(Devign_Classifier):
    def __init__(self, **kw):
        self.model = MetricLearningModel(
            input_dim=2, hidden_dim=256, aplha=0.5, lambda1=0.5,
            lambda2=0.001, dropout_p=0.2, num_layers=1
        )
        self.dataset = kw["dataset"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.item_num = kw["item_num"]
        self.epochs = kw["epoch"]
        self.learning_rate = kw["lr"]
        self.batch_size = kw["batch_size"]
        self.result_save_path = kw["outputpath"] + "/"  if kw["outputpath"][-1] != "/" else kw["outputpath"]
        model_save_path = self.result_save_path + "model/"        
        if not os.path.exists(self.result_save_path): os.makedirs(self.result_save_path)
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        self.result_save_path = self.result_save_path + str(self.item_num) + "_epo" + str(self.epochs) + "_bat" + str(self.batch_size) + "_lr" + str(self.learning_rate) + ".result"
        self.model_save_path = model_save_path + str(self.item_num) + "_bat" + str(self.batch_size) + "_lr" + str(self.learning_rate) +  "_epo"
        
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.dataset.initialize_train_batches() * self.epochs
        )
    
    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        all_steps =  self.dataset.initialize_train_batches()
        progress_bar = tqdm(range(all_steps))        
        for _ in progress_bar:
            if len(self.dataset.train_batch_indices)<=0 : continue
            self.optimizer.zero_grad()
            features, targets, same_class_features, diff_class_features = self.dataset.get_next_train_batch()
            features = features.cuda()
            targets = targets.cuda()
            same_class_features = same_class_features.cuda()
            diff_class_features = diff_class_features.cuda()
            with autocast():
                outputs, representation, loss = self.model(
                    example_batch=features, targets=targets,
                    positive_batch=same_class_features, negative_batch=diff_class_features
                )
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()    # np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()       
            
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
        all_steps =  self.dataset.initialize_test_batches()
        progress_bar = tqdm(range(all_steps))
        with torch.no_grad():
            for _ in progress_bar:
                features, targets, filename = self.dataset.get_next_test_batch()  #graph.subpdg[0].name
                features = features.cuda()
                targets = torch.LongTensor(targets).cuda()
                outputs, representation, loss = self.model(example_batch=features, targets=targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                filename_dict += list(filename)
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / (all_steps * 8)
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
        