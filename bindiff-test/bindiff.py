# coding:utf-8

import os
import time
import sqlite3

target_binary = "D:\\firmware\\tplink\\tplink_httpd.BinExport"
do_ssc_addr = 0x0407068
bindiff_exe = "D:\\software\\bindiff\\bin\\bindiff.exe"
cnt = 0

def extract_record(db_name, func_addr):
	if not os.path.exists(db_name) or not func_addr:
		return
	db = sqlite3.connect(db_name)
	cur = db.cursor()
	result = cur.execute('select address2,similarity,confidence,algorithm from function where address1="{func}";'.format(func=func_addr))
	# print(result.rowcount)
	for row in result:
		db.close()
		return row

def do_bindiff(secondary):
	global target_binary
	global bindiff_exe
	cmd = "{bindiff} --primary={primary} --secondary={secondary} --output_dir={output_dir}"
	output_dir = os.path.dirname(secondary)
	os.system(cmd.format(bindiff=bindiff_exe, primary=target_binary, secondary=secondary,output_dir=output_dir))

	bindiff_file = os.path.basename(target_binary).split(".")[0] + "_vs_" + os.path.basename(secondary).split(".")[0] + ".BinDiff"
	record = extract_record(os.path.join(output_dir, bindiff_file), do_ssc_addr)
	if record:
		log_file.write(str(record)+";"+secondary+"\n")

def walk_dir(dir_name):
	global cnt
	for path, dir_list, file_list in os.walk(dir_name):
		for f in file_list:
			file_path = os.path.join(path, f)
			if file_path.endswith(".BinExport") and (os.path.getsize(file_path) > 1024):
				cnt += 1
				do_bindiff(file_path)


def main(root_dir):
	global cnt
	l = os.listdir(root_dir)
	begin = time.time()
	# print(l)
	for dir_name in l:
		path = os.path.join(root_dir, dir_name)
		if os.path.isdir(path):
			# print("in the dir: %s" %path)
			try:
				walk_dir(path)
			except Exception as e:
				print(e)
	end = time.time()
	log_file.write("time: %s\n" %(end-begin))
	log_file.write("file: %d\n" % cnt)


if __name__ == '__main__':
	root_dir = "D:\\firmware\\bug-search\\"
	# root_dir = "D:\\firmware\\bug-search\\R6300\\squashfs-root\\"
	log_file = open(root_dir+"bindiff-spend.time", "w")
	main(root_dir)
	log_file.close()
	# extract_record("D:\\firmware\\bug-search\\R6300\\squashfs-root\\bin\\DIR_130_ssi_vs_busybox.BinDiff", do_ssc_addr)