# coding:utf-8

import os
import requests
import re
from hashlib import md5
from time import strftime, localtime
from time import sleep

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

tenda_firmwares = ["https://www.tendacn.com/en/service/download-cata-11.html"]

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def get_html(url, dst_name=None):
    log_txt = ROOT_DIR + "/log.txt"
    if os.path.exists(log_txt):
        with open(log_txt, "r") as f:
            a = re.findall(r"%s\n(.*?)\n\n" % url, f.read())
            if a:
                # print("find it in cache")
                return ROOT_DIR + "/html/" + a[0]

    # print("not in cache")
    if not dst_name:
        dst_name = md5(url.encode("utf-8")).hexdigest() + "_" + \
                   strftime("%Y%m%d%H%M", localtime()) + ".html"

    page = requests.get(url)
    if page.status_code != 200:
        print("could not get page source html: %s" % str(page.status_code))
        exit(1)

    print("download HTML page source : %s" % url)
    with open(ROOT_DIR + "/html/%s" % dst_name, "w") as f:
        f.write(page.content.decode("utf-8"))
    with open(ROOT_DIR + "/log.txt", "a") as f:
        f.write("%s\n%s\n\n" % (url, dst_name))

    # print("now, it is in cache")
    return ROOT_DIR + "/html/" + dst_name

def get_firmware():
    for url in tenda_firmwares:
        url_file = get_html(url)
        with open(url_file, "r") as ff:
            page_content = ff.read()

        firmware_pages = re.findall(r"<a href=\"//www.tendacn.com/en/download/(.*?).html\" target=\"_blank\" >", page_content)
        for page_url in firmware_pages:
            try:
                walk_page_detail("https://www.tendacn.com/en/download/"+page_url+".html")
            except Exception as e:
                undownloaded_log.write(page_url+"\n\n")
                sleep(10)
                print(e)

def download_tendalog(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                walk_page_detail(line.strip())

def walk_page_detail(url):
    url_file = get_html(url)
    with open(url_file, "r") as ff:
        page_content = ff.read()

    summary = re.findall(r'<a href="(http:|https:)?//(.*?)" target="_blank" class="downhits"', page_content)
    if (len(summary) != 1):
        undownloaded_log.write(url+"\n\n")
        return
        
    summary = summary[0]
    protocol = summary[0] if summary[0] else "https:"
    file_url = protocol + "//" + summary[1]
    filename = file_url.split("/")[-1]
    if os.path.exists(ROOT_DIR + "/tenda/" + filename):
        return 
    print("downloading %s" % file_url)
    r = requests.get(file_url, stream=True, verify=False, timeout=30)
    if not r:
        return

    with open(ROOT_DIR + "/tenda/" + filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    undownloaded_log = open(ROOT_DIR+"/tenda.log", "w")
    get_firmware()
    undownloaded_log.close()
    # download_tendalog(ROOT_DIR + "/tenda.log")