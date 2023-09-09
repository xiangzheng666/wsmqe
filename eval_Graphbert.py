from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("model/save_model/Graphbert/python_model")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model.to(device)

def comput_mrr(list_t):
    mrr = []
    for index, i in enumerate(list_t):
        i = i.cpu().numpy().tolist()
        target = i[index]
        i.sort()
        mrr.append(1 / (i.index(target) + 1))
    return np.mean(mrr)

def comput_topk(list_t, k):
    topk = []
    for index, i in enumerate(list_t):
        i = i.cpu().numpy().tolist()
        target = i[index]
        i.sort()
        if i.index(target) <= k - 1:
            topk.append(1)
        else:
            topk.append(0)
    return np.mean(topk)

language = "python"

if language == "python":
    from load_bert_dataset import get_python
    name_list = ["csnv", "DEV", "webqurey"]
    get_data_list = get_python()
else:
    from load_bert_dataset import get_java
    name_list = ["cosbeach", "csnv", "ncsed"]
    get_data_list = get_java()

from tqdm import tqdm

for name, get_data in zip(name_list, get_data_list):
    code_vect = []
    qurey_vect = []
    for code, qurey in tqdm(get_data()):
        code_input = tokenizer(code, return_tensors='pt', padding=True, truncation=True).to(device)
        qurey_input = tokenizer(qurey, return_tensors='pt', padding=True, truncation=True).to(device)

        code_output = model(**code_input)[1]  # [1] to get the CLS token representation
        qurey_output = model(**qurey_input)[1]  # [1] to get the CLS token representation

        code_vect.append(code_output)
        qurey_vect.append(qurey_output)

    code_vect = torch.stack(code_vect)
    qurey_vect = torch.stack(qurey_vect)

    scores = torch.einsum("ab,cb->ac", qurey_vect, code_vect)
    print(scores)