# coding:utf-8

import os
import re

binexport_dump = "D:\\software\\bindiff\\bin\\binexport2dump.exe"
function_algori = {
"1":	"function: name hash matching",
"2":	"function: hash matching",
"3":	"function: edges flowgraph MD index",
"4":	"function: edges callgraph MD index",
"5":	"function: MD index matching (flowgraph MD index, top down)",
"6":	"function: MD index matching (flowgraph MD index, bottom up)",
"7":	"function: prime signature matching",
"8":	"function: MD index matching (callGraph MD index, top down)",
"9":	"function: MD index matching (callGraph MD index, bottom up)",
"10":	"function: relaxed MD index matching",
"11":	"function: instruction count",
"12":	"function: address sequence",
"13":	"function: string references",
"14":	"function: loop count matching",
"15":	"function: call sequence matching(exact)",
"16":	"function: call sequence matching(topology)",
"17":	"function: call sequence matching(sequence)",
"18":	"function: call reference matching",
"19":	"function: manual",
}

def get_funcname(func_addr, file_path):
	global binexport_dump
	cmd = "{binexport_dump} {file} | findstr {func_addr}"
	f = hex(int(func_addr)).lstrip("0x").upper().rjust(8, "0")
	p = os.popen(cmd.format(binexport_dump=binexport_dump, file=file_path, func_addr=f))
	std_out = p.read()
	r = re.findall(r" [ni] (.*?)\n", std_out)
	if len(r) == 1:
		return r[0]
	else:
		print(file_path)
		print(r)
		print(f)
		exit(0)

def sort_result(raw_result, result_file):
	if not os.path.exists(raw_result):
		return None

	with open(raw_result, "r") as raw:
		items = raw.readlines()[:-2]
		# print(dir(raw))
	l = []
	for line in items:
		func_addr,sim_score,confidence,sim_alg,file_path = re.findall(r"\((\d+), (\d+\.\d+), (\d+\.\d+), (\d+)\);(.*?)\n", line)[0]
		func_name = get_funcname(func_addr, file_path)
		sim_score = sim_score[:6]
		confidence = confidence[:6]
		algorithm = function_algori[sim_alg]
		l.append((func_name, sim_score, confidence, algorithm, file_path))
	l.sort(key=lambda s : s[1])

	target_file = open(result_file, "w")
	for rank,i in enumerate(l[::-1]):
		target_file.write(str(rank)+"  "+i[0]+"  "+i[1]+"  "+i[2]+"  "+i[3]+"\n"+i[4]+"\n\n")
	target_file.close()

if __name__ == '__main__':
	root_dir = "D:\\firmware\\bug-search\\"
	sort_result(root_dir+"bindiff-spend.time", root_dir+"result.txt")