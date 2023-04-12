import os, glob, re
from parse import tokenizer
from generate_pkl import load_data, sava_data
# path = "/root/data/qm_data/issta2022/data/joren/asts/original_dataset/"
# for dir in os.listdir(path):
#     dirname = path + dir + "/"
#     print(dir)
#     for filename in os.listdir(dirname):
#         if filename[-3:] == "txt" or filename[-3:] == "dot": continue
#         for pdg in os.listdir(dirname + filename):
#             if pdg.startswith("0-"):
#                 out = dirname + filename
#                 file_path = os.path.join(out, pdg)
#                 os.system("mv "+file_path+' '+out+'.dot')
#                 os.system("rm -rf "+out)
#                 # deal_dot(out+'.dot')
#                 break

# old_ast = "/root/data/qm_data/issta2022/data/joren/pdgs/"
# new_ast = "/root/data/qm_data/issta2022/data/joren/new_pdgs/"
# for dataset_name in os.listdir(old_ast):
#     for dir_name in os.listdir(old_ast + dataset_name):
#         for file in glob.glob(old_ast + dataset_name + "/" + dir_name +"/*.dot"):
#             new_file = new_ast + dataset_name + "/" + dir_name + "/"
#             if not os.path.exists(new_file): os.makedirs(new_file)
#             new_file += file.split("/")[-1]
#             with open(file, "r") as tem_file:
#                 org_code = tem_file.readlines()
#                 nor_code = [re.sub(r"<SUB>\d+</SUB>>", "\"", i.replace("label = <(", "label = \"(")) for i in org_code]
#             tem_file.close()
#             with open(new_file, "w") as tem_file:
#                 tem_file.writelines(nor_code)
#             tem_file.close()
#             print(file)


# 处理subcode 给word2vec用
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
# a.code.apply(lambda x: tokenizer(" ".join(x), sub = False))
# a.subcode.apply(lambda x: tokenizer(" ".join(x), sub = False))

