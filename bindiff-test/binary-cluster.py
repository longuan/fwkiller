# coding:utf-8

import os, sys
import sqlite3

all_binarys = []      # 所有二进制文件的路径
binary_set = set()    # binary是否已成簇的标志
cluster_result = dict()   # 二维数组，存储最后的分簇结果

ida_path = "F:\\ida7.4\\ida.exe"
export_idc_path = "C:\\Users\\babytoy\\Desktop\\bylw\\code\\fwkiller\\bindiff-test\\export.idc"


#----------------------------------init and binexport-------------------

def do_binexport(binary_path):
	global ida_path, export_idc_path
	# ida -A -SC:\Users\babytoy\Desktop\bylw\code\fwkiller\bindiff-test\export.idc -OBinExportModule:C:\Users\babytoy\Desktop\1111.ee C:\Users\babytoy\Desktop\libc.so.1.0
	cmd = "{ida} -A -S{export_idc} -OBinExportModule:{exported_file} \"{binary}\""
	binary_path = os.path.realpath(binary_path)
	exported_file = binary_path + ".BinExport"
	if os.path.exists(exported_file):
		return exported_file
	ret = os.system(cmd.format(ida=ida_path, export_idc=export_idc_path, exported_file=exported_file, binary=binary_path))
	if ret != 0:
		print("do_binexport os.system error")
		print(binary_path)
		return None
	return exported_file

def file_filter(f, base_filename):
	if os.path.basename(f) == base_filename:
		try:
			# binwalk解包出来的软连接文件在Windows上无法打开，软连接的大小为0
			file_size = os.path.getsize(f)
		except Exception as e:
			print(e)
			return False
		if file_size < 128:
			return False
		elif file_size < 10240:
			with open(f, "r") as ff:
				# 判断文件是否为文本文件
				if "\x00" not in ff.read():
					return False
		return True
	else:
		return False


def walk_dir(firmware_name, dir_name, base_filename):
	global all_binarys
	for root_dir, _, file_list in os.walk(dir_name):
		for f in file_list:
			file_path = os.path.join(root_dir, f)
			if  file_filter(file_path, base_filename):
				exported_file = do_binexport(file_path)
				if not exported_file:
					continue
				all_binarys.append((firmware_name,exported_file))


def init(target_dir=".", base_filename="httpd"):
	# target_dir目录下每个目录是一个解包后的固件
	firmwares = os.listdir(target_dir)
	for firmware in firmwares:
		f_path = os.path.join(target_dir, firmware)
		if os.path.isdir(f_path):
			# print("in the dir: %s" %path)
			try:
				walk_dir(firmware, f_path, base_filename)
			except Exception as e:
				print(e)
				exit(1)

#----------------------------------------------------------------------------------------

#------------------------diff and extract similarity score-------------------------------

bindiff_exe = "D:\\software\\bindiff\\bin\\bindiff.exe"
def do_bindiff(primary_file, secondary_file, sec_firmware_name):
	global bindiff_exe
	output_dir = os.path.dirname(primary_file)

	pri_basename = os.path.basename(primary_file)
	pri_mainname = pri_basename[:pri_basename.rfind(".")]
	sec_basename = os.path.basename(secondary_file)
	sec_mainname = sec_basename[:sec_basename.rfind(".")]
	bindiff_file = pri_mainname + "_vs_" + sec_mainname + ".BinDiff"

	db_file = os.path.join(output_dir,pri_mainname+"_vs_"+sec_firmware_name+"_"+sec_mainname+".BinDiff")
	if os.path.exists(db_file):
		return db_file

	cmd = "{bindiff} --primary=\"{primary}\" --secondary=\"{secondary}\" --output_dir={output_dir}"
	ret = os.system(cmd.format(bindiff=bindiff_exe, primary=primary_file, secondary=secondary_file,output_dir=output_dir))
	if ret != 0:
		print("os.system error")
		print(primary_file)
		print(secondary_file)
		return None
		# exit(1)
	os.rename(os.path.join(output_dir, bindiff_file), db_file)
	return db_file

def extract_record(db_file):
	if not os.path.exists(db_file):
		return False
	db = sqlite3.connect(db_file)
	cur = db.cursor()
	num_item = cur.execute("SELECT COUNT(*) FROM function")
	# num_item = cur.execute("SELECT functions FROM file WHERE id=1")
	num_item = num_item.fetchone()[0]
	good_items = cur.execute("SELECT COUNT(*) FROM function WHERE similarity>0.89 AND confidence >0.89")
	good_items = good_items.fetchone()[0]
	db.close()
	print(good_items, num_item)
	if num_item == 0:
		return 0.0
	return good_items/num_item

def main():
	for _,target_binary in all_binarys:
		if target_binary in binary_set:
			continue
		cluster_result[target_binary] = []
		binary_set.add(target_binary)
		for secondary_firmware, secondary_binary in all_binarys:
			if secondary_binary in binary_set or target_binary == secondary_binary:
				continue
			bindiff_db = do_bindiff(target_binary, secondary_binary, secondary_firmware)
			if not bindiff_db:
				continue
			sim_score = extract_record(bindiff_db)
			if sim_score > SIMFUNC_PERCENT:
				cluster_result[target_binary].append(secondary_binary)
				binary_set.add(secondary_binary)

#----------------------------------------------------------------------

if __name__ == '__main__':
	SIMFUNC_PERCENT = float(sys.argv[1])
	log_file = sys.argv[2]
	init()
	print(len(all_binarys))
	# print(all_binarys)
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

# 注意，用os.system执行的命令不能有特殊符号：比如&和空格

# find . -name "*httpd_vs_*.BinDiff" | xargs rm
# 将httpd换为<base_filename>

# find . -name "*.pdf" -print0 | xargs -0 rm -rf
# 能够很好的找出并删除带空格的文件

# rename 's/\.extracted$//' *
# 批量去掉文件夹的.extracted

# rename 's/\s/_/g' *
# 批量将文件夹名中的空格替换为下划线

# rename 's/&/_/g' *