import numpy as np
import gc
import math

from model.CS.DeepCs.deepcs_model import deepcs
from model.CS.UNIF.UNFI_model import UNFI
from model.CS.CARLCS.carlcs_model import CARLCS

import tensorflow as tf


def comput_mrr(list_t):
    mrr = []
    for index, i in enumerate(list_t):
        i = i.numpy().tolist()
        target = i[index]
        i.sort()
        mrr.append(1 / (i.index(target) + 1))
    return np.mean(mrr), mrr


def comput_topk(list_t, k):
    topk = []
    for index, i in enumerate(list_t):
        i = i.numpy().tolist()
        target = i[index]
        i.sort()
        if i.index(target) <= k - 1:
            topk.append(1)
        else:
            topk.append(0)
    return np.mean(topk)


def compute_ndcg(list_t):
    ndcg_sum = []
    # 遍历每个查询字符串
    for query, query_relevance_annotations in enumerate(list_t):
        v = query_relevance_annotations.numpy().tolist()
        target = v[query]
        v.sort()

        dcg = v.index(target)
        query_dcg = (2 ** ((target + 1) / 2) - 1) / np.log2(dcg + 1)  # 使用该网址的相关度分数更新DCG

        ndcg_sum.append(np.clip(0.01, 1, query_dcg))
    return np.mean(ndcg_sum), ndcg_sum


def test(language, model_name):
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

    model_dict = {"deepcs": deepcs, "UNIF": UNFI, "CARLCS": CARLCS}

    model = model_dict[model_name]()
    model.load_weights("model/save_model/" + model_name + "/" + language + "/" + model_name + ".pth")

    with open("eval_output/" + language + "_" + model_name + '_QECK.txt', "w+",
              encoding="utf-8") as f:
        with open('result/QECK.txt',"w+",encoding="utf-8") as f2:
            for name, get_data in zip(name_list, get_data_list):
                mrr = []
                mrrlist = []
                ndcg=[]
                ndcglist = []
                topk = [[] for i in range(10)]
                t="model/PA/QECK/"+language+"/"+str(nd[name])+"_QECK_qurey.npy"
                for code, func, qurey in get_data(qurey_path=t):
                    tmp = model.eval([code, func, qurey])
                    value, vlist = comput_mrr(tmp)
                    ndcgvalue, ndcgvlist =compute_ndcg(tmp)
                    mrrlist.extend(vlist)
                    ndcglist.extend(ndcgvlist)
                    mrr.append(value)
                    ndcg.append(ndcgvalue)
                    tmp_topk = []
                    for k in range(10):
                        tmp_topk = comput_topk(tmp, k + 1)
                        topk[k].append(tmp_topk)
                f.write(str(mrrlist) + "\n")
                f.write(str(ndcglist) + "\n")
                print(name,":----------------------------------------------------------------------")
                f.write(name+":----------------------------------------------------------------------\n")
                print("mrr:", np.mean(mrr))
                f.write("mrr:"+ str( np.mean(mrr))+"\n")
                print("ndcg:", np.mean(ndcg))
                f.write("ndcg:"+ str(np.mean(ndcg))+"\n")
                for k in range(10):
                    print("@precison", k + 1, ":", np.mean(topk[k]))
                    f.write("@precison:" +str(k + 1)+ ":"+str(np.mean(topk[k])) + "\n")
                result={"language":language,"dataset":name,"model":model_name,"expand_method":"None",
                'mrr':np.mean(mrr),"ndcg":np.mean(ndcg),'mrr_list':vlist,"ndcg_list":ndcgvlist,'num':1,
                'topk':" ".join([str(np.mean(topk[i])) for i in range(10)])
                }
                f2.write(str(result)+'\n')
models = ["deepcs", "UNIF","CARLCS"]
language = ['java', 'python']
for l in language:
    for m in models:
        print("testing", '       ', l, '--', m, '       .............')
        test(l, m)
        gc.collect()