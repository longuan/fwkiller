# coding:utf-8


import os
import pickle
import time
import pymongo


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

IDA_PATH = "D:/tools/IDA_Pro_v7.0_Portable"
IDA_EXE  = os.path.join(IDA_PATH, "ida.exe")
IDA64_EXE = os.path.join(IDA_PATH, "ida64.exe")
script_path = os.path.join(ROOT_PATH, "extract.py")

# make sure mongo server is valid
mongo_server = pymongo.MongoClient('mongodb://admin:mima1234@172.19.218.112:27017/', serverSelectionTimeoutMS=2000)
try:
    mongo_server.server_info()
except Exception as e:
    print("cann't connect to mongo server")
    exit(0)


def load_pickle(filename):
    dumped_file = os.path.join(ROOT_PATH, "pklfiles", filename)
    with open(dumped_file, "rb") as f:
        data = pickle.load(f)
    return data


def pklfiles(firmware_name):
    dir_name = os.path.join(ROOT_PATH, "pklfiles", firmware_name)
    files = os.listdir(dir_name)
    for filename in files:
        filepath = os.path.join(dir_name, filename)
        if os.path.isfile(filepath):
            yield filepath
        else:
            print("clean ./pklfiles !!")
            continue

def dotfiles(firmware_name):
    dir_name = os.path.join(ROOT_PATH, "dotfiles", firmware_name)
    if not os.path.exists(dir_name):
        return
    files = os.listdir(dir_name)
    for filename in files:
        if not filename.endswith(".dot"):
            continue
        filepath = os.path.join(dir_name, filename)
        if os.path.isfile(filepath):
            yield filepath
        else:
            print("clean ./dotfiles !!")
            continue
# run idapython script in IDA Pro
# IDA script will dump result into pklfiles/*
def run_script(binary_path, firmware_name="default"):
    global IDA_EXE
    cmd = '{ida} -c -A -S"{script} {f}" {binary}'.format(ida=IDA_EXE,script=script_path,f=firmware_name,binary=binary_path)
    os.system(cmd)

def convert_name(src):
    # mongoDB database names cannot contain the character '.'
    return src.replace(".", "_")

def insert_data(db_name, coll_name, docu_data):
    global mongo_server

    db = mongo_server[db_name]
    coll = db[coll_name]
    for i in range(0, len(docu_data), 500):
        coll.insert_many(docu_data[i:i+500])
        # print(x.inserted_ids)


def test():
    binary_path = os.path.join(ROOT_PATH, "httpd")
    # run_script(script_path, binary_path)
    for filename in pklfiles():
        data = load_pickle(filename)
        database = mongo_server['tenda']
        coll = database['ac9']
        x = coll.insert_many(data)
        print(x.inserted_ids)
        os.remove(filename)


if __name__ == '__main__':
    main()