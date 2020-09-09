#coding: utf-8

import os
import sys
import pickle
# from preprocessing import preprocess_main,data_dir
from mytest import test
import pymongo

# 1. 遍历目录，自动化alphadiff
# 2. 解析alphadiff结果
# 3. 合并数据库

mongo_server = pymongo.MongoClient('mongodb://admin:mima1234@127.0.0.1:27017/', serverSelectionTimeoutMS=2000)
try:
    mongo_server.server_info()
except Exception as e:
    print("cann't connect to mongo server")
    exit(0)

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


def run_alphadiff(binary_a_path, binary_b_path, func_name):
    # if preprocess_main(binary_a_path, binary_b_path):
    if True:
        data_dir = "/mnt/c/Users/zzeo/Desktop/archive/alphadiff_code-ori/preprocessing_code/temp"
        result_pkl = test(data_dir, func_name)
        return result_pkl
    return None

def convert_name(src):
    # mongoDB database names cannot contain the character '.'
    return src.replace(".", "_")

db = None
def handle_result(firmware_name, binary_name, result_pkl):
    global db
    if not result_pkl:
        return
    with open(result_pkl, "rb") as f:
        data = pickle.load(f)
    result = []
    for function_name,distance in data:
        func_item = mongo_server[convert_name(firmware_name)][binary_name].find_one(
                                {"func_name":function_name})
        if not func_item:
            continue
        result.append({"func_name":function_name, "distance":round(distance,4),
                        "indegree":func_item['caller_num'], "outdegree":func_item['callee_num'],
                        "binary_name":binary_name})
    coll = db[firmware_name]
    for i in range(0, len(result), 500):
        coll.insert_many(result[i:i+500]) 


query_function = ["/mnt/c/Users/zzeo/Desktop/vmshare/firmware/_US_AC9V1.0BR_V15.03.05.16_multi_TRU01.bin.extracted/squashfs-root/bin/httpd",
                    "sub_7DBD4"]
def main():
    global db
    firmware_path = "/mnt/c/Users/zzeo/Desktop/vmshare/firmware"
    db = mongo_server[query_function[1]]
    handle_result("_US_AC10V1.0RTL_V15.03.06.23_multi_TD01.bin.extracted", "httpd", run_alphadiff(query_function[0], "/mnt/c/Users/zzeo/Desktop/vmshare/firmware/_US_AC10V1.0RTL_V15.03.06.23_multi_TD01.bin.extracted/squashfs-root/bin/httpd", query_function[1]))
    return 
    for firmware_name in os.listdir(firmware_path):
        firmware_folder = os.path.join(firmware_path, firmware_name)
        if os.path.isdir(firmware_folder):
            # print(firmware_folder)
            
            for binary_name,binary_path in walk_folder(firmware_folder):
                handle_result(firmware_name, binary_name,
                    run_alphadiff(query_function[0], binary_path, query_function[1]))
                return

if __name__ == '__main__':
    main()