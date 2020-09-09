import idc
import idaapi
import idautils
from PIL import Image
import io
import os
import math
import ast


"""
    extract machine code for a function with given start address
    Parameters:
        es, a specified function start effective-address
    Return:

"""
def extract(ea, folder_name):
    func_name  = get_func_name(ea)

    func_endEA = find_func_end(ea)

    if func_endEA == idc.BADADDR:
        print "Can't identify the end address of function " + func_name + "@" + str(hex(ea))
        return False
    
    with open(folder_name + "\\" + func_name+"_"+str(hex(ea))+".byte", "wb") as f:
        buf = get_bytes(ea, func_endEA-ea)
        byte_values = []
        for s in buf:
            byte_values.append(ord(s))
            #print str(hex(ord(s)))
        f.write(bytearray(byte_values))

        if len(byte_values) > 100*100:
            byte_values = byte_values[:10000]
        else:
            byte_values += [0x0] * (100*100 - len(byte_values))
        width = int(math.ceil(math.sqrt(func_endEA - ea)))
        # generate a gray-scale JPEG image with opcodes as pixels
        im = Image.new('L', [100, 100])
        #print "image", width, width
        im.putdata(byte_values)
        #im.show()
        im.save(folder_name + "\\" + func_name+"_"+str(hex(ea))+"_" + str(width) + "x" + str(width)+".jpeg", "JPEG", quality=100)
    
    with open(folder_name + "\\" + func_name+"_"+str(hex(ea))+".asm", "w") as f:
        E = list(FuncItems(ea))
        for e in E:
            f.write(str(hex(e)) + "\t" + GetDisasm(e) + "\n")
    
        #Exit(0)


def extract_all_functions():
    try:
        #ea = idc.ScreenEA()
        funcs = []
        for fun_ea in idautils.Functions():
            func_name = get_func_name(fun_ea)
            funcs.append((fun_ea, func_name))

        return funcs
    except Exception as e:
        print e
        return []



imported_functions = []
def imp_cb(ea, name, ord):
    if not name:
        pass
    else:
        imported_functions.append((ord, name))

    return True

def get_all_imported_functions():
    try:
        nimps = idaapi.get_import_module_qty()
        for i in xrange(0, nimps):
            idaapi.enum_import_names(i, imp_cb)
    except Exception as e:
        print e
        return []


if __name__ == "__main__":
    auto_wait()

    data_dir = idc.ARGV[1]

    is_primary = ast.literal_eval(idc.ARGV[2])
    input_path = idaapi.get_input_file_path()
    filename = input_path.split("\\")[-1]
    storage_pwd = "\\".join(input_path.split("\\")[:-1])
    folder_name = data_dir
    if is_primary:
        folder_name = os.path.join(folder_name, 'A')
    else:
        folder_name = os.path.join(folder_name, 'B')        
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    get_all_imported_functions()
    imported_names = [x for (_, x) in imported_functions]

    all_funcs = extract_all_functions()
    for ea, func_name in all_funcs:
        if func_name in imported_functions or ('.'+func_name) in imported_functions:
            continue
        extract(ea, folder_name)
    
    with open(os.path.join(storage_pwd, filename + '.imported'), 'w') as f1:
        for ordinal, func_name in imported_functions:
            f1.write(func_name + '\n')
    
    idaapi.gen_simple_call_chart(os.path.join(storage_pwd, filename),'a', r'CallGraph', idaapi.CHART_GEN_DOT)

    qexit(0)

