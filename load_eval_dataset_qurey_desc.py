import tensorflow as tf
import numpy as np

class loader():
    def __init__(self, path):
        self.real_qurey = np.load(path + "desc.npy")
        self.code = np.load(path + "code.npy")
        self.func = np.load(path + "func.npy")
        self.code_desc = np.load(path + "doc.npy")

        self.qurey = np.load(path + "doc.npy")
        self.get_qurey=False
        self.len = self.code.shape[0]
        self.mark=[0]
    def reset_qurey(self,path,get=False):
        self.get_qurey = get
        self.code_desc = np.load(path)
    def add_desc(self,pathlist,get=False):
        self.code_desc=[]
        for path in pathlist:
            self.code_desc.append(np.load(path))


def get_python_dev(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/DEV/"
    tmp = loader(path)
    yield [tmp.code, tmp.func, tmp.code_desc, tmp.real_qurey]

def get_python_webqurey(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/webqurey/"
    tmp = loader(path)

    yield [tmp.code, tmp.func, tmp.code_desc, tmp.real_qurey]



def get_python():
    return [get_python_dev,get_python_webqurey]
