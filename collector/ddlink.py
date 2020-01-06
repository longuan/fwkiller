# coding:utf-8

import os
import sys

import time  
import subprocess  
  
class TimeoutError(Exception):  
    pass  
  
def command(cmd, timeout=60):  
    """执行命令cmd，返回命令输出的内容。 
    如果超时将会抛出TimeoutError异常。 
    cmd - 要执行的命令 
    timeout - 最长等待时间，单位：秒 
    """  
    # p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)  
    p = subprocess.Popen(cmd)

    t_beginning = time.time()  
    seconds_passed = 0  
    while True:  
        if p.poll() is not None: 
            # print(p.stdout.read()) 
            break  
        seconds_passed = time.time() - t_beginning  
        if timeout and seconds_passed > timeout:  
            p.terminate()  
            raise TimeoutError(cmd, timeout)  
        time.sleep(1)  
    # return p.stdout.read()  

def read_products(l):
    with open("./dlink.product", "r") as f:
        for line in f:
            l.append(line.strip())

def run():
    l = []
    read_products(l)
    for product in l:
        try:
            command(cmd='wget -r --reject=pdf ftp://ftp2.dlink.com/PRODUCTS/%s' %product, timeout=200)
            # print("download complete: %s" %product)
        except Exception as e:
            timeout_log.write(product+"\n")
            time.sleep(60)

  
if __name__ == "__main__": 
    timeout_log = open("timeout.log", "a")
    run()
    timeout_log.close()