import os, re, pandas, pickle, glob
from collections import Counter
import pandas as pd
from clean_gadget import clean_gadget
from parse import tokenizer

def load_data(filename):
    print("reading data from: ", filename)
    f = open(filename, 'rb')
    loaded_data = pickle.load(f)
    f.close()
    return loaded_data

def sava_data(filename, data):
    print("saving data to: ", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def generate_ori_pkl_test(): # 废弃方法
    path = "./test/source_code/mutation_dataset/"
    save_path = "./test/source_code/sub_" + path.split("/")[-2] + "/"
    save_path2 = "./test/source_code/new_" + path.split("/")[-2] + "/"
    save_pkl = "./test/pkl/" + path.split("/")[-2] + "/"
    for dirname in os.listdir(path):
        print(dirname)
        final = []
        for filename in os.listdir(path + dirname):
            print(filename)
            file = path + dirname + "/" + filename
            # file = "/mnt/qm_data/issta2022/data/source_code/original_dataset/qemu/1_qemu_13033.c"
            if not os.path.exists(save_path + dirname + "/"): os.makedirs(save_path + dirname + "/")
            if not os.path.exists(save_path2 + dirname + "/"): os.makedirs(save_path2 + dirname + "/")
            if not os.path.exists(save_pkl + dirname): os.makedirs(save_pkl + dirname)
            savepath = save_path + dirname + "/" + filename
            savepath2 = save_path2 + dirname + "/" + filename
            # if os.path.exists(savepath) and os.path.exists(savepath2): continue
            try:
                f = open(file, 'r+', encoding='utf8')
                code = f.read()
            except:
                continue
            org_code = " ".join(tokenizer(code, sub = False))
            nor_code = " ".join(tokenizer(code, sub = True))

            with open(savepath, "w") as tem_file:
                tem_file.write(nor_code)
            tem_file.close()
            with open(savepath2, "w") as tem_file:
                tem_file.write(org_code)
            tem_file.close()
            final.append({"filename": filename, "code": org_code, "subcode" : nor_code, "val" : int(filename[0])})
            # print("".join(nor_code))
        
        df = pandas.DataFrame(final)
        print("saving...")
        df.to_pickle(save_pkl + dirname + "/" + dirname + ".pkl")

def generate_ori_pkl():  # 根据.c文件获取源代码和sub代码的汇总pkl
    path = "./data/source_code/mutation_dataset/"
    save_path = "./data/source_code/sub_mutation_dataset/"
    save_pkl = "./data/pkl/mutation_dataset/"
    for dirname in os.listdir(path):
        print(dirname)
        final = []
        for filename in os.listdir(path + dirname):
            print(filename)
            file = path + dirname + "/" + filename
            # file = "/mnt/qm_data/issta2022/data/source_code/original_dataset/qemu/1_qemu_13033.c"
            try:
                f = open(file, 'r+', encoding='utf8')
                code = f.read()
                code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code).strip()
            except:
                continue
            
            if not os.path.exists(save_path + dirname + "/"): os.makedirs(save_path + dirname + "/")
            if not os.path.exists(save_pkl): os.makedirs(save_pkl)
            savepath = save_path + dirname + "/" + filename
            with open(savepath, "w") as tem_file:
                tem_file.write(code)
            tem_file.close()
            with open(savepath, "r") as tem_file:
                org_code = tem_file.readlines()
                nor_code = clean_gadget(org_code)
            tem_file.close()
            with open(savepath, "w") as tem_file:
                tem_file.writelines(nor_code)
            tem_file.close()
            final.append({"filename": filename, "code": org_code, "subcode" : nor_code, "val" : int(filename[0])})
            # print("".join(nor_code))
        
        df = pandas.DataFrame(final)
        print("saving...")
        df.to_pickle(save_pkl + dirname + "/" + dirname + ".pkl")

def generate_sard_pkl():  # 根据.c文件获取源代码和sub代码的汇总pkl, 专门针对sard的处理
    path = "./data/source_code/original_dataset/sard/"
    save_path = "./data/source_code/sub_original_dataset/sard/"
    save_pkl = "./data/pkl/original_dataset/sard/"
    final = []
    for typename in os.listdir(path):
        print(typename)
        for filename in os.listdir(path + typename):
            print(filename)
            file = path + typename + "/" + filename
            try:
                f = open(file, 'r+', encoding='utf8')
                code = f.read()
                code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code).strip()
                code = code.lstrip("static ")
            except:
                continue
            
            if not os.path.exists(save_path): os.makedirs(save_path)
            if not os.path.exists(save_pkl): os.makedirs(save_pkl)
            savepath = save_path +  filename
            with open(savepath, "w") as tem_file:
                tem_file.write(code)
            tem_file.close()
            with open(savepath, "r") as tem_file:
                org_code = tem_file.readlines()
                nor_code = clean_gadget(org_code)
            tem_file.close()
            with open(savepath, "w") as tem_file:
                tem_file.writelines(nor_code)
            tem_file.close()
            final.append({"filename": filename, "code": org_code, "subcode" : nor_code, "val" : 0 if typename == "No-Vul" else 1})
            # print("".join(nor_code))
    df = pandas.DataFrame(final)
    print("saving...")
    df.to_pickle(save_pkl + "sard.pkl")

def split_data(name):  # 根据.pkl总文件分出训练集何测试机
    # path = "./data/pkl/mutation_dataset/" + name + "/" + name + "_new.pkl"
    path = "./data/pkl/original_dataset/" + name + "/" + name + "_new.pkl"
    # path = "./data/pkl/vulcnn/final_sub_original_dataset/" + name + "/" + name + ".pkl"
    save_dir = "/".join(path.split("/")[:-1]) + "/"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    df_all = load_data(path)
    seed = 0
    df_dict = {}
    kfold_num = 10
    train_dict = {i:{} for i in range(kfold_num)}
    test_dict = {i:{} for i in range(kfold_num)}
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = kfold_num, shuffle = True, random_state = seed)
    for i in Counter(df_all.val.values):
        df_dict[i] = df_all[df_all.val == i]
        for epoch, result in enumerate(kf.split(df_dict[i])):
            train_dict[epoch][i]  = df_dict[i].iloc[result[0]]
            test_dict[epoch][i] =  df_dict[i].iloc[result[1]] 
    train_all = {i:pd.concat(train_dict[i], axis=0, ignore_index=True) for i in train_dict}
    test_all = {i:pd.concat(test_dict[i], axis=0, ignore_index=True) for i in test_dict}
    sava_data(save_dir + "train.pkl", train_all)
    sava_data(save_dir + "test.pkl", test_all)

def split_data_and_repeat(name):  # 根据.pkl总文件分出训练集和测试集，并进行oversampling
    # path = "./data/pkl/original_dataset/" + name + "/" + name + "_new.pkl"
    path = "./data/pkl/vulcnn/sub_original_dataset/" + name + "/" + name + "_new.pkl"
    # path = "./data/pkl/astgru/sub_mutation_dataset/" + name + "/" + name + ".pkl"
    save_dir = "/".join(path.split("/")[:-1]) + "/"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    df_all = load_data(path)
    seed = 0
    df_dict = {}
    kfold_num = 10
    train_dict = {i:{} for i in range(kfold_num)}
    test_dict = {i:{} for i in range(kfold_num)}
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = kfold_num, shuffle = True, random_state = seed)
    for i in Counter(df_all.val.values):
        df_dict[i] = df_all[df_all.val == i]
        for epoch, result in enumerate(kf.split(df_dict[i])):
            train_dict[epoch][i]  = df_dict[i].iloc[result[0]]
            test_dict[epoch][i] =  df_dict[i].iloc[result[1]] 
    train_all = {i:pd.concat(train_dict[i], axis=0, ignore_index=True) for i in train_dict}
    test_all = {i:pd.concat(test_dict[i], axis=0, ignore_index=True) for i in test_dict}
    sava_data(save_dir + "train.pkl", train_all)
    sava_data(save_dir + "test.pkl", test_all)
    def over_sample(train_dict):
        min_num = 0 if len(train_dict[0]) < len(train_dict[1]) else 1
        num_time = len(train_dict[1-min_num]) // len(train_dict[min_num])
        num_res = len(train_dict[1-min_num])  - len(train_dict[min_num]) * num_time
        return pd.concat([train_dict[min_num]] * num_time + [train_dict[min_num].sample(n = num_res, random_state = 0)] + [train_dict[1-min_num]], axis=0, ignore_index=True)
    train_all = {i:over_sample(train_dict[i]) for i in train_dict}
    test_all = {i:pd.concat(test_dict[i], axis=0, ignore_index=True) for i in test_dict}
    sava_data(save_dir + "train_new.pkl", train_all)
    sava_data(save_dir + "test_new.pkl", test_all)

def get_txt_data():  # 获取所有代码的txt文本，用于训练sent2vec何word2vec模型
    a1 = load_data("./data/pkl/original_dataset/ffmpeg/ffmpeg.pkl")
    a2 = load_data("./data/pkl/original_dataset/qemu/qemu.pkl")
    a3 = load_data("./data/pkl/original_dataset/reveal/reveal.pkl")
    a4 = load_data("./data/pkl/mutation_dataset/ffmpeg/ffmpeg.pkl")
    a5 = load_data("./data/pkl/mutation_dataset/qemu/qemu.pkl")
    a6 = load_data("./data/pkl/mutation_dataset/reveal/reveal.pkl")
    code = "\n".join(list(a1.subcode.apply(lambda x: "".join(x))) + list(a2.subcode.apply(lambda x: "".join(x))) +\
         list(a3.subcode.apply(lambda x: "".join(x))) + list(a4.subcode.apply(lambda x: "".join(x))) +\
         list(a5.subcode.apply(lambda x: "".join(x))) + list(a6.subcode.apply(lambda x: "".join(x))))
    savepath = "./data/sub_mut_data.txt"
    with open(savepath, "w") as tem_file:
        tem_file.writelines(code)
    tem_file.close()


def generate_vulcnn_dataframe(): # vulcnn image generation的下一步处理
    input_path = "./data/joren/vulcnn_seq/sub_mutation_dataset"
    save_path = "./data/pkl/vulcnn/sub_mutation_dataset_s2v_1"
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    
    for vul_type in os.listdir(input_path):
        tem_save_path = save_path + vul_type + "/"
        if not os.path.exists(tem_save_path):
            os.makedirs(tem_save_path)
        dic = []
        dicname = input_path + vul_type
        filename = glob.glob(dicname + "/*.pkl")
        for file in filename:
            data = load_data(file)
            name = file.split("/")[-1].rstrip(".pkl")
            dic.append({
                "filename": name, 
                "length":   len(data[0]), 
                "data":     data, 
                "val":    int(name[0])})
        final_dic = pd.DataFrame(dic)
        sava_data(tem_save_path + vul_type + "_new.pkl", final_dic)


def get_word2vec_model():# 训练word2vec模型
    # a1 = load_data("./data/pkl/original_dataset/sard/sard.pkl")
    # a1 = load_data("./data/pkl/original_dataset/ffmpeg/ffmpeg_new.pkl")  # 这仨用于普通text
    # a2 = load_data("./data/pkl/original_dataset/qemu/qemu_new.pkl")
    # a3 = load_data("./data/pkl/original_dataset/reveal/reveal_new.pkl")
    # a4 = load_data("./data/pkl/mutation_dataset/ffmpeg/ffmpeg_new.pkl")  
    # a5 = load_data("./data/pkl/mutation_dataset/qemu/qemu_new.pkl")
    # a6 = load_data("./data/pkl/mutation_dataset/reveal/reveal_new.pkl")
    # a1 = load_data("./data/pkl/astgru/original_dataset/ffmpeg/ffmpeg.pkl")  # 这仨用于astgru
    # a2 = load_data("./data/pkl/astgru/original_dataset/qemu/qemu.pkl")
    # a3 = load_data("./data/pkl/astgru/original_dataset/reveal/reveal.pkl")
    a1 = load_data("./data/pkl/astgru/sub_original_dataset/ffmpeg/ffmpeg.pkl")  # 这仨用于astgru
    a2 = load_data("./data/pkl/astgru/sub_original_dataset/qemu/qemu.pkl")
    a3 = load_data("./data/pkl/astgru/sub_original_dataset/reveal/reveal.pkl")
    a4 = load_data("./data/pkl/astgru/sub_mutation_dataset/ffmpeg/ffmpeg.pkl")  # 这仨用于astgru
    a5 = load_data("./data/pkl/astgru/sub_mutation_dataset/qemu/qemu.pkl")
    a6 = load_data("./data/pkl/astgru/sub_mutation_dataset/reveal/reveal.pkl")

    from gensim.models import Word2Vec
    vector_length = 128
    # vector_length = 768
    # model = Word2Vec(list(a1.code.apply(lambda x: ("".join(x)).split()).values) + \
    #                 list(a2.code.apply(lambda x: ("".join(x)).split()).values) +  \
    #                 list(a3.code.apply(lambda x: ("".join(x)).split()).values), \
    #                 min_count=1, vector_size=vector_length, sg=1)
    # model = Word2Vec(list(a1.subcode.apply(lambda x: ("".join(x)).split()).values) + \
    #                 list(a2.subcode.apply(lambda x: ("".join(x)).split()).values) +  \
    #                 list(a3.subcode.apply(lambda x: ("".join(x)).split()).values), \
    #                 min_count=1, vector_size=vector_length, sg=1)
    # model = Word2Vec(list(a1.subcode.apply(lambda x: ("".join(x)).split()).values) , \
    #                 min_count=1, vector_size=vector_length, sg=1)
    # model = Word2Vec(list(a1.subcode) + list(a2.subcode) + list(a3.subcode) +\
    #     list(a4.subcode) + list(a5.subcode) + list(a6.subcode), \
    #     min_count=1, vector_size=vector_length, sg=1)
    model = Word2Vec(list(a1.code) + list(a2.code) + list(a3.code) + list(a6.code) + list(a4.code) + list(a5.code) , min_count=1, vector_size=vector_length, sg=1)
    import pickle
    # f = open("./data/word2vec_model/sard_subcode_" + str(vector_length) + ".pkl", 'wb')
    # f = open("./data/word2vec_model/astgru_code_" + str(vector_length) + ".pkl", 'wb')
    f = open("./data/word2vec_model/astgru_mut_subcode_" + str(vector_length) + ".pkl", 'wb')
    pickle.dump(model.wv, f)
    f.close()

def deal_subcode():# 处理subcode 给word2vec用
    a1 = load_data("./data/pkl/mutation_dataset/ffmpeg/ffmpeg.pkl")
    a2 = load_data("./data/pkl/mutation_dataset/qemu/qemu.pkl")
    a3 = load_data("./data/pkl/mutation_dataset/reveal/reveal.pkl")
    a1.code = a1.code.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("1")
    a1.subcode = a1.subcode.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("2")
    sava_data("./data/pkl/mutation_dataset/ffmpeg/ffmpeg_new.pkl", a1)
    a2.code = a2.code.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("3")
    a2.subcode = a2.subcode.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("4")
    sava_data("./data/pkl/mutation_dataset/qemu/qemu_new.pkl", a2)
    a3.code = a3.code.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("5")
    a3.subcode = a3.subcode.apply(lambda x : tokenizer(" ".join(x), sub = False))
    print("6")
    sava_data("./data/pkl/mutation_dataset/reveal/reveal_new.pkl", a3)

if __name__ == "__main__":
    # generate_ori_pkl_test()
    # generate_sard_pkl()
    # split_data("sard")
    # for name in ["ffmpeg", "qemu", "reveal"]:
    #     split_data_and_repeat(name)
    # get_txt_data()
    # generate_vulcnn_dataframe()
    get_word2vec_model()

    