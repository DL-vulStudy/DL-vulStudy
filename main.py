import os
import random
import torch
import time
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from data_loader.dataset import DataSet
from data_loader.graph_dataset import  DataSet_reveal
from model.CodeBert import CodeBert_Classifier, Bert_Classifier
from model.VulRoberta import Roberta_Classifier
from model.Code2vec import Code2vec_Classifier
from model.VulCNN import VULCNN_Classifier  # train 全靠这一个
from model.Lstm import LSTM_Classifier
from model.Gru import GRU_Classifier
from model.TextCNN import TextCNN_Classifier
from model.devign import Devign_Classifier, DevignModel
from model.reveal import Reveal_Classifier
from model.score import sava_data, load_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(cuda_num = "0"):
    seed = 2022
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

def get_model_path(model_name):
    # ["astgru", "codebert", "LSTM", "GRU", "vulcnn", "TextCNN"]
    best_result_dict = {'bert': {'ffmpeg': '4_bat64_dim128_lr3e-08_epo33.pt', 'qemu': '0_bat64_dim128_lr3e-08_epo41.pt', 'reveal': '2_bat64_dim128_lr3e-08_epo47.pt'}, 'codebert': {'ffmpeg': '3_bat64_dim128_lr2e-05_epo3.pt', 'qemu': '3_bat64_dim128_lr2e-05_epo5.pt', 'reveal': '2_bat64_dim128_lr2e-05_epo0.pt'}, 'vulcnn': {'ffmpeg': '3_bat128_dim128_lr0.0005_epo0.pt', 'qemu': '4_bat128_dim128_lr0.0005_epo6.pt', 'reveal': '3_bat128_dim128_lr0.0005_epo11.pt'}, 'astgru': {'ffmpeg': '3_bat128_dim768_lr0.0005_epo36.pt', 'qemu': '1_bat128_dim768_lr0.002_epo8.pt', 'reveal': '3_bat128_dim768_lr0.002_epo1.pt'}, 'LSTM': {'ffmpeg': '4_bat128_dim768_lr2e-05_epo3.pt', 'qemu': '2_bat128_dim768_lr2e-05_epo1.pt', 'reveal': '4_bat128_dim768_lr0.002_epo18.pt'}, 'GRU': {'ffmpeg': '4_bat128_dim768_lr0.002_epo5.pt', 'qemu': '2_bat128_dim768_lr0.002_epo43.pt', 'reveal': '3_bat128_dim768_lr0.002_epo7.pt'}, 'TextCNN': {'ffmpeg': '3_bat128_dim768_lr2e-05_epo0.pt', 'qemu': '2_bat128_dim768_lr2e-05_epo36.pt', 'reveal': '2_bat128_dim768_lr2e-05_epo7.pt'}, 'devign': {'ffmpeg': '0_bat32_lr3e-08_epo19.pt', 'qemu': '0_bat32_lr2e-05_epo28.pt', 'reveal': '0_bat32_lr3e-08_epo0.pt'}, 'reveal': {'ffmpeg': '0_bat32_lr2e-05_epo0.pt', 'qemu': '0_bat32_lr2e-05_epo23.pt', 'reveal': '0_bat32_lr2e-05_epo19.pt'}, 'vulroberta': {'ffmpeg': '3_bat64_dim768_lr0.0005_epo4.pt', 'qemu': '2_bat64_dim768_lr2e-05_epo27.pt', 'reveal': '1_bat64_dim768_lr0.0005_epo11.pt'}, 'code2vec': {'ffmpeg': '1_bat128_lr0.0005_epo12.pt', 'qemu': '2_bat128_lr0.0005_epo9.pt', 'reveal': '3_bat128_lr2e-05_epo0.pt'}}

    return {"ffmpeg": "./result/" + model_name + "/sub_original_dataset/ffmpeg/model/" + best_result_dict[model_name]["ffmpeg"],
            "qemu": "./result/" + model_name + "/sub_original_dataset/qemu/model/" + best_result_dict[model_name]["qemu"],
            "reveal": "./result/" + model_name + "/sub_original_dataset/reveal/model/" + best_result_dict[model_name]["reveal"]}

# train_new 和 test_new 指的是经过 oversampling 后的数据集
def get_kfold_dataframe(pathname = "./data/pkl/original_dataset/ffmpeg/", item_num = 0, file = "ffmpeg"):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train_final.pkl")[item_num]
    test_df = load_data(pathname + "test_final.pkl")[item_num]
    # if file == "reveal":
    #     train_df = load_data(pathname + "train_new.pkl")[item_num]
    #     test_df = load_data(pathname + "test_new.pkl")[item_num]
    # else:
    #     train_df = load_data(pathname + "train.pkl")[item_num]
    #     test_df = load_data(pathname + "test.pkl")[item_num]
    # test_df = eval_df.copy(deep=True) 
    return train_df, test_df

def get_kfold_dataframe_evaluate(pathname , item_num , file):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + file + "_new.pkl")
    test_df = train_df
    # test_df = load_data(pathname + file + "_new.pkl")
    # train_df = load_data(pathname + file + ".pkl")
    # test_df = load_data(pathname + file + ".pkl")
    return train_df, test_df


def train_code2vec(**kw):
    classifier = Code2vec_Classifier(
        file = kw["file"],
        item_num = kw["item_num"],
        epochs=kw["epoch"],
        batch_size = kw["batch_size"], 
        learning_rate = kw["lr"])
    if not kw["evaluate"]:
        datapath = "./data/pkl/code2vec/sub_original_dataset/" + kw["file"]
        outputpath = "./result/code2vec/sub_original_dataset/" + kw["file"]
        train_df = open(datapath+ "/" + kw["file"] + "_" + str(kw["item_num"]) + ".train.c2v", "r").readlines()
        test_df = open(datapath + "/" + kw["file"] + "_" + str(kw["item_num"]) + ".test.c2v", "r").readlines()
        classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath)
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "Code2vec_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("code2vec")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["sub_original_dataset", "sub_mutation_dataset"]:
                datapath = "./data/pkl/code2vec/" + dataset_type + "/" + file_val
                outputpath = "./result/code2vec/sub_original_dataset/" + kw["file"]
                if file_val == kw["file"] and dataset_type == "original_dataset":
                    train_df = open(datapath+ "/" + file_val + "_" + str(kw["item_num"]) + ".train.c2v", "r").readlines()
                    test_df = open(datapath + "/" + file_val + "_" + str(kw["item_num"]) + ".test.c2v", "r").readlines()
                else:
                    train_df = open(datapath + "/" + file_val + ".data.c2v", "r").readlines()
                    test_df = train_df
                classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath)
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result


def train_vulroberta(**kw):
    classifier = Roberta_Classifier(
            item_num = kw["item_num"], 
            epochs=kw["epoch"], 
            batch_size = kw["batch_size"], 
            learning_rate = kw["lr"])
    if not kw["evaluate"]:
        datapath = "./data/pkl/vulroberta/original_dataset/" + kw["file"]
        outputpath = "./result/vulroberta/sub_original_dataset/" + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath)
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "VulRoberta_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("vulroberta")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["original_dataset", "mutation_dataset"]:
                datapath = "./data/pkl/vulroberta/" + dataset_type + "/" + file_val
                outputpath = "./result/vulroberta/sub_original_dataset/" + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath)
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_bert(**kw):
    kw["nclass"] = 2
    kw["batch_size"] = 64
    kw["dim"] = 128
    classifier = Bert_Classifier(
            item_num = kw["item_num"], 
            epochs=kw["epoch"], 
            hidden_size = kw["dim"], 
            nclass = kw["nclass"],
            batch_size = kw["batch_size"], 
            learning_rate = kw["lr"], 
            max_len = 256)
    if not kw["evaluate"]:
        datapath = "./data/pkl/original_dataset/" + kw["file"]
        outputpath = "./result/bert/sub_original_dataset/" + kw["file"]
        if not kw["sub_or_not"]:
            outputpath = "./result/bert/sub_original_dataset/" + kw["file"]
        else:
            outputpath = "./result/bert/original_dataset/" + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath, sub_or_not = kw["sub_or_not"])
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "Bert_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("bert")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["original_dataset", "mutation_dataset"]:
                datapath = "./data/pkl/" + dataset_type + "/" + file_val
                if not kw["sub_or_not"]:
                    outputpath = "./result/bert/sub_original_dataset/" + kw["file"]
                else:
                    outputpath = "./result/bert/original_dataset/" + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath, sub_or_not = kw["sub_or_not"])
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_code_bert(**kw):
    kw["nclass"] = 2
    kw["batch_size"] = 64
    kw["dim"] = 128
    classifier = CodeBert_Classifier(
            item_num = kw["item_num"], 
            epochs=kw["epoch"], 
            hidden_size = kw["dim"], 
            nclass = kw["nclass"],
            batch_size = kw["batch_size"], 
            learning_rate = kw["lr"], 
            max_len = 256)
    if not kw["evaluate"]:
        datapath = "./data/pkl/original_dataset/" + kw["file"]
        if not kw["sub_or_not"]:
            outputpath = "./result/codebert/sub_original_dataset/" + kw["file"]
        else:
            outputpath = "./result/codebert/original_dataset/" + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath, sub_or_not = kw["sub_or_not"])
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "CodeBert_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("codebert")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["original_dataset", "mutation_dataset"]:
                datapath = "./data/pkl/" + dataset_type + "/" + file_val
                if not kw["sub_or_not"]:
                    outputpath = "./result/codebert/sub_original_dataset/" + kw["file"]
                else:
                    outputpath = "./result/codebert/original_dataset/" + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                classifier.preparation(train_df = train_df, test_df = test_df, result_save_path = outputpath, sub_or_not = kw["sub_or_not"])
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_vulcnn(**kw):
    kw["dim"] = 128
    kw["batch_size"] = 128
    classifier = VULCNN_Classifier(item_num = kw["item_num"], epochs = kw["epoch"], hidden_size = kw["dim"], batch_size = kw["batch_size"], learning_rate = kw["lr"])
    if not kw["evaluate"]:
        datapath = "./data/pkl/vulcnn/sub_original_dataset/" + kw["file"]
        outputpath = "./result/vulcnn/sub_original_dataset/" + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        classifier.preparation(
            train_df = train_df,
            test_df = test_df,
            result_save_path = outputpath
        )
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "VulCNN_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("vulcnn")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for dataset_type in["sub_original_dataset", "sub_mutation_dataset"]:
            for file_val in ["ffmpeg", "qemu", "reveal"]:
                datapath = "./data/pkl/vulcnn/" + dataset_type + "/" + file_val
                outputpath = "./result/vulcnn/sub_original_dataset/" + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "sub_original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                classifier.preparation(
                    train_df = train_df,
                    test_df = test_df,
                    result_save_path = outputpath
                )
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_text(**kw):
    kw["batch_size"] = 128
    word2vec_model = "./data/word2vec_model/ori_mut_subcode_768.pkl"
    classifier = kw["function_name"](
        max_len = 256,
        item_num = kw["item_num"],
        epochs = kw["epoch"],
        batch_size = kw["batch_size"],
        learning_rate = kw["lr"],
        hidden_size = kw["dim"],
        word2vec_model = word2vec_model
    )
    if not kw["evaluate"]:
        datapath = "./data/pkl/original_dataset/" + kw["file"]   
        outputpath = "./result/" + str(kw["function_name"]).split(".")[-1].split("_")[0] + '/sub_original_dataset/'  + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        classifier.preparation(
            train_df = train_df,
            test_df = test_df,
            result_save_path = outputpath
        )
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = str(kw["function_name"]).split(".")[-1].split("_")[0] + "_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path(str(kw["function_name"]).split(".")[-1].split("_")[0])
        print("loading model:", model_path[kw["file"]])
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["original_dataset", "mutation_dataset"]:
                datapath = "./data/pkl/" + dataset_type + "/" + file_val
                outputpath = "./result/" + str(kw["function_name"]).split(".")[-1].split("_")[0] + '/sub_original_dataset/'  + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                classifier.preparation(
                    train_df = train_df,
                    test_df = test_df,
                    result_save_path = outputpath
                )
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_astgru(**kw):
    # item_num, epoch = 100, dim = 128, batch_size = 128, function_name = GRU_Classifier, file = "ffmpeg", lr = 3e-8, evaluate = False, **kw):
    kw["batch_size"] = 128
    kw["function_name"] = GRU_Classifier
    word2vec_model = "./data/word2vec_model/astgru_mut_subcode_768.pkl"
    classifier = kw["function_name"](
        max_len=256,
        item_num = kw["item_num"],
        epochs=kw["epoch"],
        batch_size=kw["batch_size"],
        learning_rate=kw["lr"],
        hidden_size = kw["dim"],
        word2vec_model = word2vec_model
    )
    if not kw["evaluate"]:
        datapath = "./data/pkl/astgru/sub_original_dataset/" + kw["file"]
        outputpath = "./result/astgru/sub_original_dataset/"  + kw["file"]
        train_df, test_df = get_kfold_dataframe(pathname = datapath, item_num = kw["item_num"], file = kw["file"])
        train_df["subcode"] = train_df["code"] 
        test_df["subcode"] = test_df["code"] 
        classifier.preparation(
            train_df = train_df,
            test_df = test_df,
            result_save_path = outputpath
        )
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "Astgru_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("astgru")
        classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["sub_original_dataset", "sub_mutation_dataset"]:
                datapath = "./data/pkl/astgru/" + dataset_type + "/" + file_val
                outputpath = "./result/astgru/sub_original_dataset/" + kw["file"]
                function_kfold_name = get_kfold_dataframe if file_val == kw["file"] and dataset_type == "sub_original_dataset" else  get_kfold_dataframe_evaluate
                train_df, test_df = function_kfold_name(pathname = datapath, item_num = kw["item_num"], file = file_val)
                train_df["subcode"] = train_df["code"] 
                test_df["subcode"] = test_df["code"] 
                classifier.preparation(
                    train_df = train_df,
                    test_df = test_df,
                    result_save_path = outputpath
                )
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_devign(**kw):
    epoch = kw["epoch"]
    num_steps = 8
    batch_size = 32  # 8
    graph_embed_size = 200
    
    if not kw["evaluate"]:
        input_dir = "./data/pkl/devign/sub_original_dataset/" + kw["file"]
        outputpath = "./result/devign/sub_original_dataset/" + kw["file"]
        print("loading data...")
        dataset = joblib.load(open(os.path.join(input_dir, 'final_' + str(kw["item_num"]) + '.bin'), 'rb'))
        # dataset = joblib.load(open(os.path.join(input_dir, 'nvd_devign_2d_.bin'), 'rb'))
        print("loading over...")
        classifier = Devign_Classifier(dataset = dataset, graph_embed_size = graph_embed_size, 
            num_steps = num_steps, batch_size = batch_size, outputpath = outputpath, item_num = kw["item_num"], epoch = epoch, lr = kw["lr"] )
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "Devign_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("devign")
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["sub_original_dataset", "sub_mutation_dataset"]:
                datapath = "./data/pkl/devign/" + dataset_type + "/" + file_val
                outputpath = "./result/devign/sub_original_dataset/" + kw["file"]
                print("loading data...")
                if file_val == kw["file"] and dataset_type == "sub_original_dataset":
                    dataset = joblib.load(open(os.path.join(datapath, 'final_' + str(kw["item_num"]) + '.bin'), 'rb'))
                    # dataset = joblib.load(open(os.path.join(datapath, 'nvd_devign_2d_.bin'), 'rb'))
                else:
                    dataset = joblib.load(open(os.path.join(datapath, file_val + '.bin'), 'rb'))
                print("loading over...")
                classifier = Devign_Classifier(dataset = dataset, graph_embed_size = graph_embed_size, 
                    num_steps = num_steps, batch_size = batch_size, outputpath = outputpath, epoch = epoch, item_num = kw["item_num"], lr = kw["lr"] )
                classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result

def train_reveal(**kw):
    epoch = kw["epoch"]
    batch_size = 16 # 16
    graph_embed_size = 200
    
    if not kw["evaluate"]:
        input_dir = "./data/pkl/reveal/sub_original_dataset/" + kw["file"]
        outputpath = "./result/reveal/sub_original_dataset/" + kw["file"]
        print("loading data...")
        # dataset = joblib.load(open(os.path.join(input_dir, 'nvd_reveal_2d_.bin'), 'rb'))
        dataset = joblib.load(open(os.path.join(input_dir, 'final_' + str(kw["item_num"]) + '.bin'), 'rb'))
        print("loading over...")
        classifier = Reveal_Classifier(dataset = dataset, batch_size = batch_size, outputpath = outputpath, item_num = kw["item_num"], epoch = epoch, lr = kw["lr"] )
        classifier.train()
    else:
        all_result = kw["all_result"]
        row_name = "Reveal_" + kw["file"]
        if row_name not in all_result: all_result[row_name] = {}
        model_path = get_model_path("reveal")
        for file_val in ["ffmpeg", "qemu", "reveal"]:
            for dataset_type in["sub_original_dataset", "sub_mutation_dataset"]:
                datapath = "./data/pkl/reveal/" + dataset_type + "/" + file_val
                outputpath = "./result/reveal/sub_original_dataset/" + kw["file"]
                if file_val == kw["file"] and dataset_type == "sub_original_dataset":
                    dataset = joblib.load(open(os.path.join(datapath, 'final_' + str(kw["item_num"]) + '.bin'), 'rb'))
                    # dataset = joblib.load(open(os.path.join(datapath, 'nvd_reveal_2d_.bin'), 'rb'))
                else:
                    dataset = joblib.load(open(os.path.join(datapath, file_val + '.bin'), 'rb'))
                classifier = Reveal_Classifier(dataset = dataset, batch_size = batch_size, outputpath = outputpath, item_num = kw["item_num"], epoch = epoch, lr = kw["lr"] )
                classifier.model.load_state_dict(torch.load(model_path[kw["file"]]))
                start = time.time()
                _, score_dict = classifier.eval()
                end = time.time()
                score_dict["time"] = format(end-start, '.3f')
                col_name = "ori" + "_" + file_val if "ori" in dataset_type else "mut" + "_" + file_val
                all_result[row_name][col_name] = score_dict
        all_result_pd = pd.DataFrame(all_result)
        sava_data("./result/all_result.pkl", all_result_pd)
        sava_data("./result/all_result_dict.pkl", all_result)
        return all_result


def devign_prepare(filename, i): # 如果要生成测试数据集的话，就把注释的替换
    batch_size = 8
    node_tag = "node_features"
    graph_tag = "graph"
    label_tag = "target"
    # input_dir = "./data/pkl/devign/sub_mutation_dataset/" + filename
    input_dir = "./data/pkl/devign/sub_original_dataset/" + filename
    processed_data_path = os.path.join(input_dir, "final_" + str(i) + ".bin")
    # processed_data_path = os.path.join(input_dir, 'nvd_devign_2d_.bin')
    # processed_data_path = os.path.join(input_dir, filename + '.bin')
    if True and os.path.exists(processed_data_path):
        print('file already exist')
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_' + str(i) + '.txt'),
        # dataset = DataSet(train_src=os.path.join(input_dir, 'nvd_train.txt'),
        # dataset = DataSet(train_src=None,
                            valid_src=None,
                            test_src=os.path.join(input_dir, 'test_' + str(i) + '.txt'),
                            # test_src=os.path.join(input_dir, 'nvd_test.txt'),
                            # test_src=os.path.join(input_dir, filename + '.txt'),
                            batch_size=batch_size, n_ident=node_tag, g_ident=graph_tag,
                            l_ident=label_tag)
        file = open(processed_data_path, 'wb')
        joblib.dump(dataset, file)
        file.close()

def reveal_prepare(filename, i): # 如果要生成测试数据集的话，就把注释的替换
    batch_size = 16
    # input_dir = "./data/pkl/reveal/sub_mutation_dataset/" + filename
    input_dir = "./data/pkl/reveal/sub_original_dataset/" + filename
    processed_data_path = os.path.join(input_dir, "final_" + str(i) + ".bin")
    # processed_data_path = os.path.join(input_dir, 'nvd_reveal_2d_.bin')
    # processed_data_path = os.path.join(input_dir, filename + '.bin')
    if True and os.path.exists(processed_data_path):
        print('file already exist')
    else:
        dataset = DataSet_reveal(train_src=os.path.join(input_dir, 'train_' + str(i) + '.txt'),
                            test_src=os.path.join(input_dir, 'test_' + str(i) + '.txt'),
        # dataset = DataSet_reveal(train_src=os.path.join(input_dir, 'nvd_train.txt'),
        #                     test_src=os.path.join(input_dir, 'nvd_test.txt'),
        # dataset = DataSet_reveal(train_src=None,
        #                     test_src=os.path.join(input_dir, filename + '.txt'),
                            batch_size=batch_size)
        file = open(processed_data_path, 'wb')
        joblib.dump(dataset, file)
        file.close()


def save_ggnn(model, dataset, save_path):
    cnt=1
    #batch_num=len(dataset.train_batches)
    #try:
    while len(dataset.test_batches) !=0 :
        model.eval()
        model.zero_grad()
        graph, targets = dataset.get_next_test_batch()
        targets = targets.long()
        targets = torch.LongTensor(targets).cuda()
        outputs = model(graph, cuda=True)
        outputs = outputs.detach().cpu().tolist()
        temp1=0
        for pdg in graph.subpdg:
            temp2 = pdg.graph.number_of_nodes()
            all_features = outputs[temp1:temp1+temp2]
            sum_feature = outputs[0]
            for feature in all_features[1:]:
                sum_feature=list(np.add(sum_feature, feature))
            res_target=[]
            # if pdg.label == 0:
            #     res_target=[1,0]
            # else:
            #     res_target=[0,1]
            json_dict = { 
            'graph_feature':sum_feature,
            #'target':res_target
            'target':pdg.label
            }
            temp1=temp1+temp2
            
            file_path = os.path.join(save_path, pdg.name)
            with open(file_path, 'w', encoding='utf-8') as fp:
                json.dump(json_dict, fp)
            print(cnt)
            cnt+=1

def reveal_save_ggnn(filename):
    input_dir = "./data/pkl/devign/sub_mutation_dataset/" + filename
    # input_dir = "./data/pkl/devign/sub_original_dataset/" + filename
    save_path = "/root/data/qm_data/issta2022/data/devign&reveal/Reveal-Recurrence-main/Vuld_SySe/save_ggnn/sub_mutation_dataset/"+ filename
    # save_path = "/root/data/qm_data/issta2022/data/devign&reveal/Reveal-Recurrence-main/Vuld_SySe/save_ggnn/sub_original_dataset/"+ filename
    if not os.path.exists(save_path): os.makedirs(save_path)
    # processed_data_path = os.path.join(input_dir, 'nvd_devign_2d_.bin')
    processed_data_path = os.path.join(input_dir, filename + '.bin')
    print("reading from", processed_data_path)
    dataset = joblib.load(open(processed_data_path, 'rb'))
    model = DevignModel(input_dim=dataset.feature_size, output_dim=200,num_steps=8, max_edge_types=4)
    model.cuda()
    save_ggnn(model=model, dataset=dataset,save_path=save_path)


def prepare_devign_reveal():
    for i in range(10):
        for file in ["ffmpeg", "qemu", "reveal"]:
            devign_prepare(file, i)
            reveal_prepare(file, i)
            reveal_save_ggnn(file)  # 先用save_ggnn获得reveal的数据，再用prepare获得dataset


def main():
    epoch = 50
    dim = 768
    evaluate = False
    sub_or_not = False
    set_seed()
    for item_num in range(5):
        for lr in [2e-5, 3e-8, 0.0005]:
        # for lr in [2e-5, 3e-8, 0.002, 0.0005]:
        # for lr in [2e-5, 0.002]:
        # for lr in [3e-8]:
            # for file in ["ffmpeg"]:
            for file in ["ffmpeg", "qemu", "reveal"]:
                if item_num == 0 :continue
                if item_num == 1 and file == "ffmpeg" :continue
                # train_bert(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_code_bert(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_text(item_num = item_num, dim = dim, function_name = LSTM_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_text(item_num = item_num, dim = dim, function_name = GRU_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_text(item_num = item_num, dim = dim, function_name = TextCNN_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_vulcnn(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_astgru(item_num = item_num, dim = dim, file = file, epoch = epoch, lr = lr, evaluate = evaluate, sub_or_not = sub_or_not)
                # train_vulroberta(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, batch_size = 64)
                # train_code2vec(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, batch_size = 128)
                train_devign(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr)
                # train_reveal(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr)
                

def main_test():
    epoch = 100
    lr  = 2e-5
    dim = 768
    evaluate = True
    sub_or_not = False
    set_seed()
    item_num = 0
    # all_result = {}
    all_result = load_data("./result/all_result_dict.pkl")
    for file in ["ffmpeg", "qemu", "reveal"]:
        # all_result = train_bert(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_code_bert(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_text(item_num = item_num, dim = dim, function_name = LSTM_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_text(item_num = item_num, dim = dim, function_name = GRU_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_text(item_num = item_num, dim = dim, function_name = TextCNN_Classifier, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_vulcnn(item_num = item_num, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_astgru(item_num = item_num, dim = dim, file = file, epoch = epoch, lr = lr, evaluate = evaluate, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_vulroberta(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, batch_size = 64, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_devign(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, all_result = all_result, sub_or_not = sub_or_not)
        # all_result = train_reveal(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, all_result = all_result, sub_or_not = sub_or_not)
        all_result = train_code2vec(item_num = item_num, file = file, evaluate = evaluate, epoch = epoch, lr = lr, batch_size = 128, all_result = all_result, sub_or_not = sub_or_not)
        
        


if __name__ == "__main__":
    # prepare_devign_reveal()
    # main()
    main_test()
    
    
