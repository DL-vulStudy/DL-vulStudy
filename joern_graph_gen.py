#encoding=utf-8
from logging import root
import os, re
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess


def get_all_file(path):
    path = path[0]
    file_list = []
    path_list = os.listdir(path)
    for path_tmp in path_list:
        full = path + path_tmp + '/'
        for file in os.listdir(full):
            file_list.append(file)
    return file_list

def deal_dot(path):
    with open(path, "r") as tem_file:
        org_code = tem_file.readlines()
        nor_code = [re.sub(r"<SUB>\d+</SUB>>", "\"", i.replace("label = <(", "label = \"(")) for i in org_code]
    tem_file.close()
    nor_code
    with open(path, "w") as tem_file:
        tem_file.writelines(nor_code)
    tem_file.close()

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str, default='/home/survey_devign/survey_data_preprocess/novul_bin')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str, default='/home/survey_devign/survey_data_preprocess/novul_output_pdg')
    parser.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r', '--repr', help='The type of representation: pdg or lineinfo_json', type=str, default='pdg')
    args = parser.parse_args()
    return args

def joern_parse(file, outdir):
    record_txt =  os.path.join(outdir,"parse_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch "+record_txt)
    with open(record_txt,'r') as f:
        rec_list = f.readlines()
    name = file.split('/')[-1].split('.')[0]
    if name+'\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ',name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    os.environ['file'] = str(file)
    os.environ['out'] = str(out) #parse后的文件名与source文件名称一致
    os.system('sh joern-parse $file --language c -o $out')
    with open(record_txt, 'a+') as f:
        f.writelines(name+'\n')

def joern_export(bin, outdir, repr):
    record_txt =  os.path.join(outdir,"export_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch "+record_txt)
    with open(record_txt,'r') as f:
        rec_list = f.readlines()

    name = bin.split('/')[-1].split('.')[0]
    out = os.path.join(outdir, name)
    if name+'\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ',name)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    
    if repr in ['pdg', "ast"]:
        os.system('sh joern-export $bin'+ " --repr " + repr + ' -o $out') # cpg 改成 pdg
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("1-"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv "+file_path+' '+out+'.dot')
                    os.system("rm -rf "+out)
                    deal_dot(out+'.dot')
                    break
        except:
            pass
    else:
        pwd = os.getcwd()
        if out[-4:] != 'json':
            out += '.json'
        joern_process = subprocess.Popen(["./joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        import_cpg_cmd = f"importCpg(\"{bin}\")\r"
        script_path = f"{pwd}/graph-for-funcs.sc"
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{out}\"\r" #json
        cmd = import_cpg_cmd + run_script_cmd
        ret , err = joern_process.communicate(cmd)
        print(ret,err)

    len_outdir = len(glob.glob(outdir + '*'))
    print('--------------> len of outdir ', len_outdir)
    with open(record_txt, 'a+') as f:
        f.writelines(name+'\n')


def main(input_path, output_path, type, repr):
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'
    if not os.path.exists(output_path): os.makedirs(output_path)

    pool_num = 16
        
    pool = Pool(pool_num)

    if type == 'parse':
        # files = get_all_file(input_path)
        files = glob.glob(input_path + '*.c')
        pool.map(partial(joern_parse, outdir = output_path), files)

    elif type == 'export':
        bins = glob.glob(input_path + '*.bin')
        if repr in ['pdg', "ast"]:
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
        else:
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)

    else:
        print('Type error!')    

if __name__ == '__main__':
    joern_path = '/root/data/qm_data/devign/joern/joern-cli/'
    os.chdir(joern_path)
    input_root_path = "/root/data/qm_data/issta2022/test/source_code/"
    output_bin_path = "/root/data/qm_data/issta2022/test/joren/bins/"
    output_pdg_path = "/root/data/qm_data/issta2022/test/joren/pdgs/"
    output_ast_path = "/root/data/qm_data/issta2022/test/joren/asts/"
    to_test_dir = ["sub_original_dataset"]
    # to_test_dir = ["new_original_dataset", "sub_original_dataset"]
    for dir in to_test_dir:
        for dir_name in os.listdir(input_root_path + dir):
            main(input_root_path + dir + "/" + dir_name , output_bin_path + dir+ "/" + dir_name, "parse", "pdg") 
            main(output_bin_path + dir + "/" + dir_name , output_pdg_path + dir+ "/" + dir_name, "export", "pdg") 
            main(output_bin_path + dir + "/" + dir_name , output_ast_path + dir+ "/" + dir_name, "export", "ast") 
