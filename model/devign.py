

import torch, os
from tqdm import tqdm
from dgl.nn import GatedGraphConv
from torch import nn
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as f
from torch_geometric.nn import GatedGraphConv as GatedGraphConv_2
from torch_geometric.nn import global_mean_pool
from model.Lstm import LSTM_Classifier
from model.score import get_MCM_score, sava_data, load_data
from transformers import AdamW, get_linear_schedule_with_warmup


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types=4, num_steps=6):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim #100
        self.out_dim = output_dim #200
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.batchnorm_1d = torch.nn.BatchNorm1d(output_dim)
        self.batchnorm_1d_for_concat = torch.nn.BatchNorm1d(self.concat_dim)

        #self.dropout_1d = torch.nn.Dropout(0.5)
        #self.dropout_1d_for_concat = torch.nn.Dropout(0.5)
        
        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        graph = graph.to(torch.device("cuda:0"))
        features = features.to(torch.device("cuda:0"))
        edge_types = edge_types.to(torch.device("cuda:0"))
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                #self.conv_l1(h_i.transpose(1, 2))
                self.batchnorm_1d(
                    self.conv_l1(h_i.transpose(1, 2)) #outputs
                )
                
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                #self.conv_l2(Y_1)
                self.batchnorm_1d(
                    self.conv_l2(Y_1) #outputs
                )
                
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                #self.conv_l1_for_concat(c_i.transpose(1, 2))
                self.batchnorm_1d_for_concat(
                    self.conv_l1_for_concat(c_i.transpose(1, 2)) #ouputs+feature
                )
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                #self.conv_l2_for_concat(Z_1)
                self.batchnorm_1d_for_concat(   
                    self.conv_l2_for_concat(Z_1) #ouputs+feature
                )
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        #before_avg = torch.mul(self.dropout_1d(self.mlp_y(Y_2)), self.dropout_1d_for_concat(self.mlp_z(Z_2)))
        avg = before_avg.mean(dim=1)
        #return avg
        #result = self.sigmoid(avg).squeeze(dim=-1)
        result = self.sigmoid(avg)
        #result = self.softmax(avg)
        return result


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device("cuda:0"))
        features = features.to(torch.device("cuda:0"))
        edge_types = edge_types.to(torch.device("cuda:0"))
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result


class Devign_simplify(nn.Module):
    def __init__(self,output_dim, max_edge_types=4, num_steps=6):
        super(DevignModel, self).__init__()
        self.out_dim = output_dim #200
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.relu = nn.ReLU()
        self.ggnn = GatedGraphConv_2(out_channels=output_dim,
                                   num_layers=num_steps)
        self.classifier = nn.Linear(in_features=output_dim, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        #graph, features, edge_types = batch.get_network _inputs().to(torch.decive("cuda:0"))
        x = x.to(torch.device("cuda:0"))
        edge_index = edge_index.to(torch.device("cuda:0"))
        outputs = self.ggnn(x, edge_index)
        outputs = self.relu(outputs)
        pooled = global_mean_pool(outputs, torch.zeros(outputs.shape[0], dtype=int, device=outputs.device))
        avg = self.classifier(pooled)
        result = self.sigmoid(avg)
        return result



class Devign_Classifier(LSTM_Classifier):
    def __init__(self, **kw):
        self.model = DevignModel(input_dim=kw["dataset"].feature_size, output_dim=kw["graph_embed_size"],
                        num_steps=kw["num_steps"], max_edge_types=4)
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
            num_training_steps=self.dataset.initialize_train_batch() * self.epochs
        )
    
    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        all_steps =  self.dataset.initialize_train_batch()
        progress_bar = tqdm(range(all_steps))        
        for _ in progress_bar:
            self.optimizer.zero_grad()
            graph, targets = self.dataset.get_next_train_batch()
            targets = targets.long()
            targets = torch.LongTensor(targets).cuda()
            with autocast():
                outputs = self.model(graph, cuda=True)
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
        all_steps =  self.dataset.initialize_test_batch()
        progress_bar = tqdm(range(all_steps))
        with torch.no_grad():
            for _ in progress_bar:
                graph, targets = self.dataset.get_next_test_batch()  #graph.subpdg[0].name
                filename = [i.name.split(".")[0] for i  in graph.subpdg]
                targets = targets.long()
                targets = torch.LongTensor(targets).cuda()
                outputs = self.model(graph, cuda=True)
                loss = self.loss_fn(outputs, targets)
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
        