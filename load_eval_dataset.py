import tensorflow as tf
import numpy as np
pool_size=1000
class loader():
    def __init__(self, path):
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

def get_java_cosbeach(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/CosBeach/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path,list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path,get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc[:pool_size],tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_java_csnv(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/CSN-V/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:

            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_java_ncsed(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/NCSED/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_java_transfer(qurey_path=None,get_qurey=False):
    path= "dataset/eval_dataset/java/transfer/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_python_csnv(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/CSN-V/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_python_dev(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/DEV/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_python_webqurey(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/webqurey/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_python_transfer(qurey_path=None,get_qurey=False):
    path = "dataset/eval_dataset/python/transfer/"
    tmp = loader(path)
    if qurey_path:
        if isinstance(qurey_path, list):
            tmp.add_desc(qurey_path, get_qurey)
        else:
            tmp.reset_qurey(qurey_path, get_qurey)
    if get_qurey:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:], tmp.qurey[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:],
                   tmp.qurey[0-len(tmp.code_desc)%pool_size:,:]]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc,tmp.qurey]
    else:
        for i in range(len(tmp.code_desc)//pool_size):
            yield [tmp.code[i*pool_size:(i+1)*pool_size,:], tmp.func[i*pool_size:(i+1)*pool_size,:], tmp.code_desc[i*pool_size:(i+1)*pool_size,:]]
        if len(tmp.code_desc)%pool_size != 0 and len(tmp.code_desc) > pool_size:
            yield [np.concatenate([tmp.code[0-len(tmp.code_desc)%pool_size:,:],tmp.code[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   np.concatenate([tmp.func[0-len(tmp.code_desc)%pool_size:,:],tmp.func[:pool_size-len(tmp.code_desc)%pool_size,:]]), 
                   tmp.code_desc[0-len(tmp.code_desc)%pool_size:,:]
                   ]
        elif len(tmp.code_desc) < pool_size :
            yield [tmp.code[:pool_size],tmp.func[:pool_size],tmp.code_desc]

def get_java():
    return [get_java_cosbeach,get_java_csnv,get_java_ncsed,get_java_transfer]

def get_python():
    return [get_python_csnv,get_python_dev,get_python_webqurey,get_python_transfer]
