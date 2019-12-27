# coding:utf-8

import queue
from ftplib import FTP
import os

class MyFtp:
    def __init__(self, host, port=21):
        self.ftp = FTP()
        self.ftp.connect(host, port)

    def login(self, username, pwd):
        self.ftp.set_debuglevel(1)  # 打开调试级别2，显示详细信息
        self.ftp.login(username, pwd)

    def downloadFile(self, localpath, filename):
        file_handle = open(localpath + filename, "wb")  # 以写模式在本地打开文件
        self.ftp.retrbinary('RETR %s' % os.path.basename(filename), file_handle.write, blocksize=1024)  # 下载ftp文件
        file_handle.close()

    def getPwd(self):
        return self.ftp.pwd()

    def cwd(self, dirname):
        self.ftp.cwd(dirname)

    def close(self):
        self.ftp.set_debuglevel(0)  # 关闭调试
        self.ftp.quit()

class D_LINK:
    def __init__(self):
        self.host = "ftp2.dlink.com"
        self.port = 21
        self.username = "anonymous"
        self.password = ""
        self.ftp_conn = MyFtp(self.host, self.port)
        self.ftp_conn.login(self.username, self.password)
        self.cwd_list = queue.Queue()
        self.cwd_list.put("/PRODUCTS")
        self.log_file = open("./dlink.log", "w")
        self.firmware_path = "../firmware/"

    def download_files(self, pwd):
        filelist = []
        self.ftp_conn.ftp.retrlines('LIST .', filelist.append)
        for line in filelist:
            if line.startswith("d"):
                # append directory name into self.cwd_list
                self.cwd_list.put(pwd+'/'+line.split(" ")[-1])
            elif line.startswith("-"):
                # this is a file
                filename = line.split(" ")[-1]
                file_ext = filename.split(".")[-1]
                if file_ext.upper() in ('ZIP', 'BIN', 'HEX'):
                    if os.path.exists(self.firmware_path+filename):
                        continue
                    self.ftp_conn.downloadFile(self.firmware_path, filename)

    def walk(self):
        while not self.cwd_list.empty():
            dir_name = self.cwd_list.get()
            self.ftp_conn.cwd(dir_name)
            self.download_files(dir_name)
            # print("walking in %s" % dir_name)

    def __del__(self):
        self.ftp_conn.close()
        self.log_file.close()

if __name__ == '__main__':
    dlink = D_LINK()
    dlink.walk()
