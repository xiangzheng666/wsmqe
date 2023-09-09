import os
import tensorflow as tf
import numpy as np

l="python"

code_tokens_id2word = {}
code_tokens_word2id = {}
with open("../train_data/"+l+"/code_tokens_vocab.txt", "r+", encoding='utf-8') as f:
    for index, i in enumerate(f.read().split("\n")):
        code_tokens_id2word[index] = i
        code_tokens_word2id[i] = index

docstring_tokens_id2word = {}
docstring_tokens_word2id = {}
with open("../train_data/"+l+"/docstring_tokens_vocab.txt", "r+", encoding='utf-8') as f:
    for index, i in enumerate(f.read().split("\n")):
        docstring_tokens_id2word[index] = i
        docstring_tokens_word2id[i] = index

func_name_id2word = {}
func_name_word2id = {}
with open("../train_data/"+l+"/func_name_vocab.txt", "r+", encoding='utf-8') as f:
    for index, i in enumerate(f.read().split("\n")):
        func_name_id2word[index] = i
        func_name_word2id[i] = index

cloums=["qurey_filter.txt","code_filter.txt","desc_filter.txt","func_filter.txt"]
dicts={"code_filter.txt":code_tokens_word2id,"qurey_filter.txt":docstring_tokens_word2id,"desc_filter.txt":docstring_tokens_word2id,"func_filter.txt":func_name_word2id}
length={"code_filter.txt":200,"qurey_filter.txt":60,"desc_filter.txt":60,"func_filter.txt":10}
np_name={"code_filter.txt":"code.npy","qurey_filter.txt":"doc.npy","func_filter.txt":"func.npy",'desc_filter.txt':'desc.npy'}

if l=="python":
    dir=["transfer"]
if l=="java":
    dir = ["transfer"]
num=0
tmp=[]
for i in dir:
    for j in cloums:
        path = os.path.join(os.path.join(l, i), j)
        dictionary=dicts[j]
        if os.path.exists(path):
            with open(path,"r+", encoding='utf-8') as f:
                tmp_data=[[dictionary[word] for word in line.split(" ")] for line in f.read().split("\n")]
                if j == "qurey_filter.txt":
                    tmp=[index for index,i in enumerate(tmp_data) if len(i)<6]
                data = [tmp_data[i] for i in tmp]
                num= len(data)
                print(num)
                data=tf.keras.preprocessing.sequence.pad_sequences(data,
                                                              maxlen=length[j], padding='post',
                                                              truncating='post', value=0)
                np.save(os.path.join(os.path.join(l, i),np_name[j]),data)
        else:
            data = [[] for i in range(num)]
            data = tf.keras.preprocessing.sequence.pad_sequences(data,
                                                                 maxlen=length[j], padding='post',
                                                                 truncating='post', value=0)
            np.save(os.path.join(os.path.join(l, i), np_name[j]), data)

