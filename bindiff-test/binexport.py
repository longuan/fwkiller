# coding:utf-8

import os

ida_path = "F:\\ida7.4\\ida.exe"
export_idc_path = "C:\\Users\\babytoy\\Desktop\\bylw\\code\\fwkiller\\bindiff-test\\export.idc"

def do_export(binary):
	# ida -A -SC:\Users\babytoy\Desktop\bylw\code\fwkiller\bindiff-test\export.idc -OBinExportModule:C:\Users\babytoy\Desktop\1111.ee C:\Users\babytoy\Desktop\libc.so.1.0
	cmd = "{ida} -A -S{export_idc} -OBinExportModule:{exported_file} {binary}"
	binary_path = os.path.realpath(binary)
	exported_file = binary_path + ".BinExport"
	os.system(cmd.format(ida=ida_path, export_idc=export_idc_path, exported_file=exported_file, binary=binary_path))


if __name__ == '__main__':
	do_export("C:\\Users\\babytoy\\Desktop\\libc.so.1.0")