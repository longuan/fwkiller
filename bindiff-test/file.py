# coding:utf-8

import os
from binexport import do_export

def is_binary(file_path):
	if not os.path.exists(file_path):
		return True
	if not os.path.getsize(file_path):
		return False

	filename = os.path.basename(file_path)
	if "." in filename and "so" not in filename:
		return False

	if "idb" in filename or "BinExport" in filename:
		return False

	with open(file_path, "rb") as f:
		content = f.read(8192)
		if b'\0' not in content:
			return False

	return True

def reorganize(dir_name):
	for path, dir_list, file_list in os.walk(dir_name):
		if not dir_list and not file_list:
			os.rmdir(path)

		for f in file_list:
			file_path = os.path.join(path, f)
			if not is_binary(file_path):
				try:
					os.remove(file_path)
				except Exception as e:
					print(e)
			else:
				if not os.path.exists(file_path+".BinExport"):
					do_export(file_path)

def main(root_dir):
	l = os.listdir(root_dir)
	# print(l)
	for dir_name in l:
		path = os.path.join(root_dir, dir_name)
		if os.path.isdir(path):
			print("in the dir: %s" %path)
			reorganize(path)

if __name__ == '__main__':
	main("D:\\firmware\\bug-search\\")