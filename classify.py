#coding:utf-8

# 根据出度入度，以及alphadiff的相似结果进行预分类，看看效果


import os
import sys
import math
import networkx as nx
from main import *


def walk_folder(firmware_folder):

    for dir_name, dirs, files in os.walk(firmware_folder):
        for binary_name in files:
            binary_path = os.path.join(dir_name, binary_name)
            if not os.path.exists(binary_path):
                continue
            if os.path.getsize(binary_path) <= 1024:
                continue
            if ".dot" in binary_name:
                continue
            if ".id" in binary_name or ".nam" in binary_name or ".til" in binary_name:
                continue
            if "." in binary_name and ".so" not in binary_name and ".ko" not in binary_name:
                continue
            yield (binary_name,binary_path)


def handle_firmware():
    for firmware_name in os.listdir(firmware_path):
        firmware_folder = os.path.join(firmware_path, firmware_name)
        if os.path.isdir(firmware_folder):
            print(firmware_folder)

            db_name = convert_name(firmware_name)

            for binary_name,binary_path in walk_folder(firmware_folder):
                print(binary_name, end=" ")
                run_script(binary_path, firmware_name)
            print()

            # for pkl_file in pklfiles(firmware_name):
            #     table_name = convert_name(os.path.basename(pkl_file))
            #     data = load_pickle(pkl_file)
            #     insert_data(db_name, table_name, data)
            #     os.remove(pkl_file)
            for dot_path in dotfiles(firmware_name):
                result = []
                dot_file = os.path.basename(dot_path)
                table_name = os.path.splitext(dot_file)[0]
                import_file = table_name + '.import'
                import_path = os.path.join(ROOT_PATH, "pklfiles",firmware_name, import_file)
                import_funcs = load_pickle(import_path)
                callgraph = nx.drawing.nx_agraph.read_dot(dot_path)
                
                for node in callgraph.nodes:
                    node_name = callgraph._node[node]['label']
                    if node_name in import_funcs:
                        continue
                    in_degree = callgraph.in_degree(node)
                    out_degree = callgraph.out_degree(node)
                    result.append({"func_name":node_name, "callee_num":out_degree, "caller_num":in_degree})
                insert_data(db_name, table_name, result)
                # os.remove(dot_path)
                # os.remove(import_path)
            
target_indegree = 1
target_outdegree = 3
other_target = {"_US_AC10V1_0RTL_V15_03_06_23_multi_TD01_bin_extracted":['setQosMiblist', 12, 1, 4816596], 
                "_US_AC15V1_0BR_V15_03_05_18_multi_TD01_bin_extracted":['sub_7DEC0', 12,1, 515776],
                "_US_AC15V1_0BR_V15_03_1_17_multi_TD01_bin_extracted":['sub_6DB8C', 7,1,449420],
                "_US_AC15V1_0BR_V15_03_1_12_multi_TD01_bin_extracted":['sub_6C98C', 7,1,444812],
                "_US_AC18V1_0BR_V15_03_3_10_multi_TD01_bin_extracted":["sub_77C08", 8,1,490504]}

def pre_classify():
    function_all_count = 0
    binary_all_count   = 0
    firmware_all_count = 0
    supposed_count     = 0

    result = dict()
    suppose_indegree = (int(target_indegree*0.85), int(math.ceil(target_indegree*1.25)))
    suppose_outdegree = (int(target_outdegree*0.85), int(math.ceil(target_outdegree*1.25)))
    dblist = mongo_server.list_database_names()
    for db_name in dblist:
        result[db_name] = list()
        db = mongo_server[db_name]
        firmware_all_count += 1
        for coll_name in db.list_collection_names():
            coll = db[coll_name]
            binary_all_count += 1
            function_all_count += coll.count_documents({})
            for indegree in range(suppose_indegree[0], suppose_indegree[1]+1):
                records = coll.find({"caller_num":indegree}, {"_id":0})
                for x in records:
                    if suppose_outdegree[0]<=x['callee_num']<=suppose_outdegree[1]:
                        result[db_name].append(
                            [coll_name, x['func_name'],x['callee_num'], x['caller_num']])

    with open("inout_classify_result-new3.txt", "w") as f:
        f.write("firmware all count:"+str(firmware_all_count-3)+"\n")
        f.write("binary all count:"+str(binary_all_count)+"\n")
        f.write("function all count:"+str(function_all_count)+"\n")
        for _,value in result.items():
            supposed_count += len(value)
        f.write("suppose all count:"+str(supposed_count)+"\n")

        for key,value in result.items():
            f.write(key+":"+str(len(value))+"\n")
        f.write("\n")
        for key,value in result.items():
            f.write(key+"\n"+str(value)+"\n\n")

def delete_repeat_data():
    for func_name in coll.distinct('func_name'):
        num = coll.count_documents({"func_name":func_name})
        if num > 1:
            for _ in range(1, num):
                coll.delete_one({"func_name":func_name})

if __name__ == '__main__':
    # firmware_path = sys.argv[1]
    firmware_path = r"C:\Users\zzeo\Desktop\vmshare\firmware"
    pre_classify()
