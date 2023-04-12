import os, torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from model.VulCNN import VULCNN_Classifier
from model.score import get_MCM_score, sava_data, load_data
from transformers import RobertaConfig, RobertaTokenizerFast, BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

class CodeBertDataset(Dataset):
  def __init__(self, **kw):
    self.data = kw["data"]
    # self.texts = texts
    # self.targets = targets
    self.tokenizer =  kw["tokenizer"]
    self.max_len = kw["max_len"]
    self.sub_or_not = kw["sub_or_not"]
    

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    text = "".join(self.data.subcode.iloc[index]) if not self.sub_or_not else "".join(self.data.code.iloc[index])
    target = self.data.val.iloc[index]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'filename':self.data.filename.iloc[index].split(".")[0],
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

class Ori_CodeBert(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.embedding_dim = 768
        self.hidden_dim = 128
        self.dropout = config.hidden_dropout_prob
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            # nn.Dropout(self.dropout),
            #nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_labels)
        )
        # self.classifier_lstm = TextRNNAtten(config)
        self.init_weights()
        

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits, last_hidden_state_cls
    
    def get_mean_forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        return mean_output

class CodeBert_Classifier(VULCNN_Classifier):
    def __init__(self, **kw):
        self.config = RobertaConfig.from_pretrained('microsoft/codebert-base', num_labels = kw["nclass"])
        self.model = Ori_CodeBert.from_pretrained('microsoft/codebert-base', config=self.config)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.max_len = kw["max_len"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        
        self.item_num = kw["item_num"]
        self.hidden_size = kw["hidden_size"]


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
        self.train_set = CodeBertDataset(data = self.train_df, tokenizer = self.tokenizer, max_len = self.max_len, sub_or_not = kw["sub_or_not"])
        self.test_set = CodeBertDataset(data = self.test_df, tokenizer = self.tokenizer, max_len = self.max_len, sub_or_not = kw["sub_or_not"])

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
        # self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)
        # self.valid_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)

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
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)
                filename = data["filename"]
                outputs, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
    
    # def train(self, Load_Pretrained=False):
    #     best_accuracy = 0
    #     learning_record_dict = {}
    #     start = 0
    #     train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
    #     test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
    #     if Load_Pretrained == True and os.path.exists(self.result_save_path + ".result"):
    #         learning_record_dict = load_data(self.result_save_path + ".result")
    #         start = len(learning_record_dict)
    #         for i in learning_record_dict:
    #             train_table.add_row(["tra", str(i+1), format(learning_record_dict[i]["train_loss"], '.4f')] + \
    #                 [learning_record_dict[i]["train_score"][j] for j in learning_record_dict[i]["train_score"] if j != "MCM"])
    #             test_table.add_row(["val", str(i+1), format(learning_record_dict[i]["val_loss"], '.4f')] + \
    #                 [learning_record_dict[i]["val_score"][j] for j in learning_record_dict[i]["val_score"] if j != "MCM"])
    #     for epoch in range(start, self.epochs):
    #         print(f'Epoch {epoch + 1}/{self.epochs}')
    #         train_loss, train_score = self.fit()
    #         train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])
    #         print(train_table)

    #         val_loss, val_score = self.eval()
    #         test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
    #         print(test_table)
            
        #     if float(val_score["M_f1"]) > best_accuracy:
        #         torch.save(self.model.state_dict(), self.model_save_path)
        #         best_accuracy = val_loss
            
        #     learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
        #         "train_score": train_score, "val_score": val_score}
        #     sava_data(self.result_save_path + ".result", learning_record_dict)
        #     print("\n")

        # if self.epochs == 0:
        #     torch.save(self.model.state_dict(), self.model_save_path )
        # else:
        #     torch.save(self.model.state_dict(), self.model_save_path + "-every")
        # self.model.load_state_dict(torch.load(self.model_save_path))


class Bert_Classifier(CodeBert_Classifier):
    def __init__(self, **kw):
        self.config = BertConfig.from_pretrained("bert-base-uncased", num_labels=kw["nclass"])
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_len = kw["max_len"]
        self.epochs = kw["epochs"]
        self.batch_size = kw["batch_size"]
        self.learning_rate = kw["learning_rate"]
        self.model.to(self.device)
        
        self.item_num = kw["item_num"]
        self.hidden_size = kw["hidden_size"]