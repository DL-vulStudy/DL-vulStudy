import networkx as nx
from anytree import AnyNode
import argparse
import os
import pandas as pd
import glob
from multiprocessing import Pool
from functools import partial
import  pickle

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

def parse_options():
    parser = argparse.ArgumentParser(description='transform ast to sequence.')
    parser.add_argument('-d', '--dir', help='The path of a dir which consists of some dot_files', required=True)
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    args = parser.parse_args()
    return args


def createtree(tree, dad_child, lable_code, nodelist=None, lable=None, parent=None):
    id = len(nodelist)

    if lable not in nodelist:
        try:
            child = dict(dad_child[lable])
            a = lable_code[lable].split('"(')[1].split(')"')[0]
            #n = len(a.split(','))
            t = a.split(',')[0]
            #d = a.split(',')[1]
        except IndexError:
            t = ''
            #d = ''
            #n = 2
        if id == 0:
            tree.token = t
            #tree.data = d
            #tree.num = n
        else:
            newnode = AnyNode(id=id, token=t, parent=parent)#, data=d num=n,
        nodelist.append(lable)
        for c in child:
            if id == 0:
                createtree(tree, dad_child, lable_code, nodelist, c, parent=tree)
            else:
                createtree(tree, dad_child, lable_code, nodelist, c, parent=newnode)


def allseq(dot, out_path, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    print(dot_name)

    if dot_name not in existing_files:
        try:
            ast = nx.drawing.nx_pydot.read_dot(dot)
            adj_dict = dict(ast.adj)
            labels_dict = nx.get_node_attributes(ast, 'label')

            nodelist = []
            newtree = AnyNode(id=0, token=None)#, data=None, num=None

            root = list(adj_dict.keys())[0]
            createtree(newtree, adj_dict, labels_dict, nodelist, root)

            seq = []
            sequence(newtree, seq)
            res = {"filename": dot_name, "code" : seq, "val" : int(dot_name[0])}
            out_put = out_path + dot_name + '.pkl'
            sava_data(out_put, res)
        except:
            print("ERROE")
    

def sequence(tree, s):
    child = tree.children
    #if tree.num == 2:
    #    s.append(tree.token)
    #else:
    #    if tree.data != '' and tree.data != '<empty>':
    #        s.append(tree.data)
    #    else:
    s.append(tree.token)

    for c in child:
        sequence(c, s)


def main(file):
    # args = parse_options()
    # dir_name = args.dir
    # out_path = args.out
    dir_name = "/root/data/qm_data/issta2022/data/joren/new_asts/sub_mutation_dataset/" + file
    out_path = "/root/data/qm_data/issta2022/data/joren/seqs/sub_mutation_dataset/" + file
    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += '/'

    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'

    if not os.path.exists(out_path): os.makedirs(out_path)
    existing_files = os.listdir(out_path)
    existing_files = [f.split('.txt')[0] for f in existing_files]

    dotfiles = glob.glob(dir_name + '*.dot')
    # dic = []
    # for dot in dotfiles:
    #     res = allseq(dot, out_path=out_path, existing_files=existing_files)
    #     if res != None:
    #         dic.append(res)
    # final_dic = pd.DataFrame(dic)
    # sava_data(out_path + file + ".pkl", final_dic)

    pool = Pool(32)
    pool.map(partial(allseq, out_path=out_path, existing_files=existing_files), dotfiles)

def generate_pkl(file):
    in_path = "/root/data/qm_data/issta2022/data/joren/seqs/original_dataset/" + file
    out_path = "/root/data/qm_data/issta2022/data/pkl/astgru/original_dataset/" + file
    in_path = in_path + "/" if in_path[-1] !="/" else in_path
    out_path = out_path + "/" if out_path[-1] !="/" else out_path
    if not os.path.exists(out_path): os.makedirs(out_path)
    dotfiles = glob.glob(in_path + '*.pkl')
    dic = []
    for dot in dotfiles:
        dic.append(load_data(dot))
    final_dic = pd.DataFrame(dic)
    sava_data(out_path + file + "_new.pkl", final_dic)

if __name__ == '__main__':
    for file in ["ffmpeg", "qemu", "reveal"]:
        # main(file)
        generate_pkl(file)
