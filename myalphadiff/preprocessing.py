import os
import sys
import networkx as nx

# Need To re-configure!!!
idaw_path = "D:\\tools\\IDA_Pro_v7.0_Portable\\idat.exe"
extract_imported_and_cg_script = "D:\\GraduationProject\\fwkiller\\myalphadiff\\extract_imported_and_cg_ida7.py"
data_dir = 'D:\\GraduationProject\\fwkiller\\myalphadiff\\temp'

def extract_imported_and_callgraph(bin_path, data_path, is_primary):
    cmd = idaw_path + " -c -A -S\"" + extract_imported_and_cg_script + " " + str(data_path) + \
           " "  + str(is_primary) + "\" "  + bin_path
    print(cmd)
    os.system(cmd)

def generate_signature(dotpath, imported_funcs_self, common_imported, bindiff, tag):
    callgraph = nx.drawing.nx_agraph.read_dot(dotpath)

    #callseq_layer1 = []
    result = {}
    for node in callgraph.nodes:
        node_name = callgraph._node[node]['label']
        node_name = node_name[1:] if node_name[0] == '.' else node_name
        if node_name in imported_funcs_self:
            continue
        signature = {}

        in_degree = callgraph.in_degree(node)
        out_degree = callgraph.out_degree(node)
        signature['in_degree'] = in_degree
        signature['out_degree'] = out_degree


        callseq = []
        for succ in callgraph.successors(node):
            succ_name = callgraph._node[succ]['label']
            succ_name = succ_name[1:] if succ_name[0] == '.' else succ_name
            #print 'successor', succ, succ_name
            if succ_name in common_imported:
                #callseq_layer1[common_imported[succ_name]] = 1
                callseq.append(common_imported[succ_name])
                #print 'imported succ', node, node_name, in_degree, out_degree, succ_name, common_imported[succ_name]
                #break
        signature['callseq'] = callseq
        result[node_name] = signature
        #print node, node_name, in_degree, out_degree, callseq_layer1
        #break
    # output to file
    fd_path = os.path.join(data_dir, str(bindiff), tag)
    if not os.path.isdir(fd_path):
        os.makedirs(fd_path)
    with open(os.path.join(fd_path, 'signature'), 'w') as f:
        f.write(str(len(common_imported)) + '\n')
        for key in result.keys():
            sig = result[key]
            f.write(key + ':' + str(sig['in_degree']) + ' ' + str(sig['out_degree']) + '##' + str(sig['callseq']) + '\n')
    
    return True


def preprocess_main(path_unstripped_A, path_unstripped_B):
    extract_imported_and_callgraph(path_unstripped_A, data_dir, True)
    if not os.path.isfile(path_unstripped_A.split('.idb')[0] + '.dot'):
        print ("dot does not exist", path_unstripped_A.split('.idb')[0] + '.dot')
        #exit(0)
        return False

    extract_imported_and_callgraph(path_unstripped_B, data_dir, False)
    if not os.path.isfile(path_unstripped_B.split('.idb')[0] + '.dot'):
        return False

    imported_A = []
    imported_B = []

    # imported functions of pre file
    with open(path_unstripped_A + '.imported', 'r') as f:
        for line in f:
            if line is None or line == '':
                continue
            #print line[:-1]
            imported_A.append(line[:-1])

    # imported functions of post file
    with open(path_unstripped_B + '.imported', 'r') as f:
        for line in f:
            if line is None or line == '':
                continue
            #print line[:-1]
            imported_B.append(line[:-1])
    
    # common imported function
    common_imported = {}
    index = 0
    for func_name in imported_A:
        if func_name in imported_B:
            func_name = func_name[1:] if func_name[0] == '.' else func_name
            common_imported[func_name] = index
            index = index + 1

    # generate context signature for pre file
    if not generate_signature(path_unstripped_A+'.dot', imported_A, common_imported, '', 'A'):
        return False

    # generate context signature for post file
    if not generate_signature(path_unstripped_B+'.dot', imported_B, common_imported, '', 'B'):
        return False
    return True


if __name__ == "__main__":
    binary_a = sys.argv[1]
    binary_b = sys.argv[2]

    assert(os.name=="nt" and sys.platform=="win32")
    os.system("rmdir /s/q "+data_dir)
    os.system("md "+data_dir)

    main(binary_a, binary_b, data_dir)
