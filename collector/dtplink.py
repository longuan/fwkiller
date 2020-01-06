# coding:utf-8

import os
import requests
import re
from hashlib import md5
from time import strftime, localtime

http_header = {
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
    "cookie":"_ga=GA1.2.1150960249.1577933311; _gid=GA1.2.1093159465.1577933311; _fbp=fb.1.1577933311526.159976641; accepted_local_switcher=1; _gat_global=1; _gat=1"
}

Tenda_cve_urls = [
    "https://www.cvedetails.com/vulnerability-list/vendor_id-13620/Tenda.html"
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

    page = requests.get(url, headers=http_header)
    if page.status_code != 200:
        print("!!!!!")
        print(url)
        print("could not get page source html: %s" % str(page.status_code))
        exit(1)

    print("download HTML page source : %s" % url)
    with open(ROOT_DIR + "/html/%s" % dst_name, "w") as f:
        f.write(page.content.decode("utf-8"))
    with open(ROOT_DIR + "/log.txt", "a") as f:
        f.write("%s\n%s\n\n" % (url, dst_name))

    # print("now, it is in cache")
    return ROOT_DIR + "/html/" + dst_name



def get_all_product():
    url = "https://www.tp-link.com/us/support/download/"
    url_file = get_html(url)
    with open(url_file, "r") as ff:
        page_content = ff.read()
    products = re.findall(r'class="ga-click" data-ga="Support-Download-Center-(.*?)" href="(.*?)" target="_blank">', page_content)
    for product in products:
        download_firmware(product[1])
    # print(products)

def download_firmware(product):
    url = "https://www.tp-link.com/" + product
    url_file = get_html(url)
    with open(url_file, "r") as ff:
        page_content = ff.read()
    
    if "Please choose hardware version:" in page_content:
        hardware_versions = re.findall(r'<li data-value="(.*?)"', page_content)
        for hardware_version in hardware_versions:
            url_file = get_html(url+hardware_version.lower()+"/#Firmware")
            with open(url_file, "r") as ff:
                page_content = ff.read()
            firmware_addr = re.findall(r"data-ga='Download-Detail-Firmware-(.*?)' target=\"_blank\" href=\"(.*?)\"", page_content)
            if(len(firmware_addr) == 0):
                return 
            else:
                for f in firmware_addr:
                    download_file(ROOT_DIR+"/tplink/"+f[1].split("/")[-1], f[1])
    # TODO: downlaod else:

def download_file(localpath, remotepath):
    if os.path.exists(localpath):
        return
    r = requests.get(remotepath, stream=True, verify=False, timeout=30)
    if not r:
        return

    print("downloading %s" % localpath)
    with open(localpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    get_all_product()