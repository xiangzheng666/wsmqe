import os

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

import re

def tokenit(txt):
    return [re.sub("Ġ","",i).lower() for i in tokenizer.tokenize(txt)]

l="python"

cloums=["code.txt","qurey.txt","func.txt"]


if l=="python":
    dir=["DEV"]
if l=="java":
    dir = ["CSN-V", "CosBeach", "NCSED","transfer"]

vocab={"code.txt":"code_tokens_vocab.txt","qurey.txt":"docstring_tokens_vocab.txt","func.txt":"func_name_vocab.txt","desc.txt":"docstring_tokens_vocab.txt"}
ada={"code.txt":"code_filter.txt","qurey.txt":"qurey_filter.txt","func.txt":"func_filter.txt","desc.txt":"desc_filter.txt"}
for wen in dir:
    for j in cloums:
        path = os.path.join(os.path.join(l, wen), j)
        if os.path.exists(path):
            with open(path, "r+", encoding="utf-8") as f:
                data = f.read().split("\n")
    
            with open("../train_data/"+l+"/"+vocab[j], "r+", encoding="utf-8") as f:
                codevocab = f.read().split("\n")
                tokern2id = {codevocab[i]: i for i in range(len(codevocab))}
    
            out=[]
            for i in data:
                tmp=[]
                for i in tokenit(i):
                    if i in tokern2id.keys():
                        tmp.append(i)
                    else:
                        tmp.append("<oov>")
                out.append(" ".join(tmp))
    
            with open(l+"/"+wen+"/"+ada[j], "w+", encoding="utf-8") as f:
                f.write("\n".join(out))