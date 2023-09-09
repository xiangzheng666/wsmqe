import numpy as np
import gc
import math

import tensorflow as tf

def count_trailing_zeros(lst):
    count = 0
    for i in range(len(lst)-1, -1, -1):
        if lst[i] == 0:
            count += 1
        else:
            break
    return count

def test(language):
    if language == "python":
        from load_eval_dataset import get_python
        name_list = ["csnv", "DEV", "webqurey", "transfer"]
        nd = {"DEV": "DEV", "csnv": "CSN-V", "webqurey": "webqurey","transfer":"transfer"}
        get_data_list = get_python()
    else:
        from load_eval_dataset import get_java
        name_list = ["cosbeach", "csnv", "ncsed", "transfer"]
        nd = {"cosbeach": "CosBeach", "csnv": "CSN-V", "ncsed": "NCSED","transfer":"transfer"}
        get_data_list = get_java()
    tmp=[]
    for name, get_data in zip(name_list, get_data_list):
        result={}
        size=[]
        for code, func, qurey in get_data():
            for i in qurey:
                size.append(60-count_trailing_zeros(i))
        result['language']=language
        result['dataset']=name
        result['length']=size
        tmp.append(result)
    return tmp
language = ['java', 'python']
with open('result/data_length.txt',"w+",encoding="utf-8") as f2:
    for l in language:
        print("======================",l,"======================")
        result = test(l)
        f2.write("\n".join([str(i ) for i in result])+'\n')
        gc.collect()