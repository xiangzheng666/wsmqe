import tensorflow as tf
import numpy as np

class loader():
    def read(self,path):
        with open(path,'r',encoding='utf-8') as f:
            data=f.read().split('\n')
        if len(data)>1000:
            return data[:1000]
        else:
            return data
    def __init__(self, path):
        self.code = self.read(path+"code.txt")
        self.code_desc = self.read(path+"qurey.txt")

def get_java_cosbeach(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/CosBeach/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc
def get_java_csnv(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/CSN-V/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc
def get_java_ncsed(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/NCSED/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc

def get_python_csnv(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/CSN-V/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc
def get_python_dev(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/DEV/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc
def get_python_webqurey(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/webqurey/"
    tmp = loader(path)
    yield tmp.code,tmp.code_desc

def get_java():
    return [get_java_cosbeach,get_java_csnv,get_java_ncsed]

def get_python():
    return [get_python_csnv,get_python_dev,get_python_webqurey]
