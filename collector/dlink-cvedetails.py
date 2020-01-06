# coding:utf-8

import os
import requests
import re
from hashlib import md5
from time import strftime, localtime

DLink_cve_urls = [
    "https://www.cvedetails.com/vulnerability-list.php?vendor_id=9740&page=1&hasexp=0&opdos=0&opec=0&opov=0&opcsrf=0&opgpriv=0&opsqli=0&opxss=0&opdirt=0&opmemc=0&ophttprs=0&opbyp=0&opfileinc=0&opginf=0&cvssscoremin=0&cvssscoremax=0&year=0&month=0&cweid=0&order=1&trc=110&sha=bc84e95fc401bce64401934a02e15dd6163f9d48",
    "https://www.cvedetails.com/vulnerability-list.php?vendor_id=9740&product_id=&version_id=&page=2&hasexp=0&opdos=0&opec=0&opov=0&opcsrf=0&opgpriv=0&opsqli=0&opxss=0&opdirt=0&opmemc=0&ophttprs=0&opbyp=0&opfileinc=0&opginf=0&cvssscoremin=0&cvssscoremax=0&year=0&month=0&cweid=0&order=1&trc=110&sha=bc84e95fc401bce64401934a02e15dd6163f9d48",
    ""
]

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

def walk_Dlink_urls():
    for url in DLink_cve_urls:
        url_file = get_html(url)
        with open(url_file, "r") as ff:
            page_content = ff.read()

        cve_numbers = re.findall(r"title=\"(.*?) security vulnerability details\"", page_content)
        for cve_number in cve_numbers:
            walk_cve_detail("https://www.cvedetails.com/cve/"+cve_number)

def walk_cve_detail(url):
    url_file = get_html(url)
    with open(url_file, "r") as ff:
        page_content = ff.read()

    summary = re.findall(r'<div class="cvedetailssummary">\s*(.*?)\s*<span class="datenote">', page_content)
    dlink_products = re.findall(r"title=\"Product Details Dlink (.*?) Firmware\"", page_content, re.I)

    dlink_cve_txt.write(url+"\n"+summary[0]+"\n"+" ".join(dlink_products)+"\n\n")

if __name__ == "__main__":
    dlink_cve_txt = open(ROOT_DIR + "/dlink_cve.txt", "w")
    walk_Dlink_urls()
    dlink_cve_txt.close()