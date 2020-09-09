# coding:utf-8


import os
import idautils
import ida_segment
import ida_nalt
import pickle

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DOTS_PATH = os.path.join(ROOT_PATH, "dotfiles")
PKLS_PATH = os.path.join(ROOT_PATH, "pklfiles")

output_log = open(os.path.join(ROOT_PATH, "output.log"), "w")

text_seg = ida_segment.get_segm_by_name(".text")
if not text_seg:
    text_seg = ida_segment.get_segm_by_name("LOAD")

def log(s):
    if is_batch_mode:
        output_log.write(s+"\n")
    else:
        print s

def dump_pickle(result):
    binary_name = get_root_filename()
    dumped_file = os.path.join(fw_pkl, binary_name)
    with open(dumped_file, "wb") as f:
        pickle.dump(result, f, -1)

def dump_import(result):
    binary_name = get_root_filename()
    filename = os.path.splitext(binary_name)[0] + ".import"
    dumped_file = os.path.join(fw_pkl, filename)
    with open(dumped_file, "wb") as f:
        pickle.dump(result, f, -1)


def get_all_callees(ea, s):
    pass

def get_all_callers(func_ea, s):
    pass

def get_importfunc_refs(ea):
    func_names = set()
    for ins in CodeRefsTo(ea,0):
        func_names.add(get_func_name(ins))

    print(len(func_names))

def get_callee(ea, s):
    """如果ea指令是函数调用指令，记录此指令调用的函数，先不管是text段函数还是导入函数"""
    for callee in idautils.CodeRefsFrom(ea, 0):
        f_name = get_func_name(callee)
        if f_name != get_func_name(ea):
            s.add(f_name)

def get_caller(func_ea, s):
    """func_ea是函数的首地址。记录所有调用func_ea函数的那些函数 """
    for cg_in in CodeRefsTo(func_ea, 0):
        # if text_seg.start_ea <= cg_in <= text_seg.end_ea:
        if is_code(get_full_flags(cg_in)):
            s.add(get_func_attr(cg_in, FUNCATTR_START))


def get_all_importfunc():
    nimps = ida_nalt.get_import_module_qty()
    result = []
    # print("Found %d import(s)..." % nimps)
    for i in range(nimps):
        name = ida_nalt.get_import_module_name(i)
        if not name:
            # print("Failed to get import module name for #%d" % i)
            # name = "<unnamed>"
            pass

        # print("Walking imports for module %s" % name)
        def imp_cb(ea, name, ordinal):
            if name:
                # print("%08x: %s (ordinal #%d)" % (ea, name, ordinal))
                result.append(name)
            # True -> Continue enumeration
            # False -> Stop enumeration
            return True
        ida_nalt.enum_import_names(i, imp_cb)
    dump_import(result)

# func_name | callers | callees | import_funcs

def dump_callgraph(dot_name):
    dot_path = os.path.join(fw_dot, dot_name)
    idaapi.gen_simple_call_chart(dot_path,'a', r'CallGraph', idaapi.CHART_GEN_DOT|idaapi.CHART_NOLIBFUNCS)

def extract_CG_old():
    result = list()
    for function in Functions(text_seg.start_ea, text_seg.end_ea):
        func_name = get_func_name(function)
        f = idaapi.get_func(function)

        callers = set()
        get_caller(function, callers)

        callees = set()
        flowchart = idaapi.FlowChart(f)
        for bb in flowchart:
            for ins_addr in Heads(bb.start_ea, bb.end_ea):
                get_callee(ins_addr, callees)

                # 对每条指令做处理
                # ins = idautils.DecodeInstruction(ins_addr)
                # if ins == None:
                #     continue
        result.append({"func_name":func_name,"func_startea":function,
                        "callers": list(callers), "callees":list(callees)})
    # 将result dump到pkl_path文件夹下
    dump_pickle(result)

def main(batch_mode):
    """ batch_mode是指ida pro的命令行自动化模式，此模式下不易调试脚本 """
    if batch_mode:
        auto_wait()
        dump_callgraph(get_root_filename())
        get_all_importfunc()
        output_log.close()
        qexit(0)
    else:
        output_log.close()
        dump_callgraph(get_root_filename())
        get_all_importfunc()

if __name__ == '__main__':
    is_batch_mode = True if len(idc.ARGV)>1 else False
    if is_batch_mode:
        firmware_name = idc.ARGV[1]
    else:
        firmware_name = "default"
    fw_dot = os.path.join(DOTS_PATH, firmware_name)
    if not os.path.exists(fw_dot):
        os.mkdir(fw_dot)
    fw_pkl = os.path.join(PKLS_PATH, firmware_name)
    if not os.path.exists(fw_pkl):
        os.mkdir(fw_pkl)
    main(is_batch_mode)
