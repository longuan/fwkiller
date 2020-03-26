# coding:utf-8

import os, sys
import sqlite3

all_binarys = []      # 所有二进制文件的路径
binary_set = set()    # binary是否已成簇的标志
cluster_result = dict()   # 二维数组，存储最后的分簇结果

ida_path = "F:\\ida7.4\\ida.exe"
export_idc_path = "C:\\Users\\babytoy\\Desktop\\bylw\\code\\fwkiller\\bindiff-test\\export.idc"

def do_binexport(binary_path):
	global ida_path, export_idc_path
	# ida -A -SC:\Users\babytoy\Desktop\bylw\code\fwkiller\bindiff-test\export.idc -OBinExportModule:C:\Users\babytoy\Desktop\1111.ee C:\Users\babytoy\Desktop\libc.so.1.0
	cmd = "{ida} -A -S{export_idc} -OBinExportModule:{exported_file} {binary}"
	binary_path = os.path.realpath(binary_path)
	exported_file = binary_path + ".BinExport"
	if os.path.exists(exported_file):
		return exported_file
	os.system(cmd.format(ida=ida_path, export_idc=export_idc_path, exported_file=exported_file, binary=binary_path))
	return exported_file


def walk_dir(dir_name, base_filename):
	global all_binarys
	for root_dir, _, file_list in os.walk(dir_name):
		for f in file_list:
			file_path = os.path.join(root_dir, f)
			if f == "httpd":
				exported_file = do_binexport(file_path)
				all_binarys.append(exported_file)


def init(target_dir=".", base_filename="httpd"):
	# target_dir目录下每个目录是一个解包后的固件
	firmwares = os.listdir(target_dir)
	for firmware in firmwares:
		f_path = os.path.join(target_dir, firmware)
		if os.path.isdir(f_path):
			# print("in the dir: %s" %path)
			try:
				walk_dir(f_path, base_filename)
			except Exception as e:
				print(e)
				exit(1)


bindiff_exe = "D:\\software\\bindiff\\bin\\bindiff.exe"
def do_bindiff(primary_file, secondary_file):
	global bindiff_exe
	output_dir = os.path.dirname(primary_file)

	# TODO: 此处需要改进
	sec_firmware = os.path.dirname(secondary_file).split("\\")[-1]
	db_file = os.path.join(output_dir,sec_firmware+".BinDiff")
	if os.path.exists(db_file):
		return db_file

	cmd = "{bindiff} --primary={primary} --secondary={secondary} --output_dir={output_dir}"
	os.system(cmd.format(bindiff=bindiff_exe, primary=primary_file, secondary=secondary_file,output_dir=output_dir))

	pri_basename = os.path.basename(primary_file)
	sec_basename = os.path.basename(secondary_file)
	bindiff_file = pri_basename[:pri_basename.rfind(".")] + "_vs_" + sec_basename[:sec_basename.rfind(".")] + ".BinDiff"

	os.rename(os.path.join(output_dir, bindiff_file), db_file)
	return db_file

def extract_record(db_file):
	if not os.path.exists(db_file):
		return False
	db = sqlite3.connect(db_file)
	cur = db.cursor()
	num_item = cur.execute("SELECT COUNT(*) FROM function")
	num_item = num_item.fetchone()[0]
	good_items = cur.execute("SELECT COUNT(*) FROM function WHERE similarity>0.89 AND confidence >0.9")
	good_items = good_items.fetchone()[0]
	db.close()
	print(good_items, num_item)
	return good_items/num_item

def main():
	for target_binary in all_binarys:
		if target_binary in binary_set:
			continue
		cluster_result[target_binary] = []
		binary_set.add(target_binary)
		for secondary_binary in all_binarys:
			if secondary_binary in binary_set or target_binary == secondary_binary:
				continue
			bindiff_db = do_bindiff(target_binary, secondary_binary)
			sim_score = extract_record(bindiff_db)
			if sim_score > 0.9:
				cluster_result[target_binary].append(secondary_binary)
				binary_set.add(secondary_binary)

if __name__ == '__main__':
	log_file = sys.argv[1]
	init()
	print(all_binarys)
	main()
	with open(log_file, "w") as f:
		f.write("summary\n")
		f.write("\tthe number of binarys: "+str(len(all_binarys))+"\n")
		f.write("\tthe number of clusters:"+str(len(cluster_result))+"\n\n")
		for k,v in cluster_result.items():
			f.write(k+"\n")
			for i in v:
				f.write("\t"+i+"\n")
			f.write("\n")