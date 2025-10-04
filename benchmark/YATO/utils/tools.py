# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:33
# @Author  : Peilin Zhou
# @FileName: tools.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import os
import json
def check_dir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
def save_params(params):
    output={}
    final_params = {}
    for key,value in params.items():
        if key!='device' and isinstance(value,str) or isinstance(value,int) or isinstance(value,float):
            final_params[key] = value
    output["params"]=final_params
    json_str = json.dumps(output, indent=4)
    with open(params['model_dir']+'params.json', 'w') as json_file:
        json_file.write(json_str)
def save_eval(entity_level_dic,token_level_dic,data):
    output ={}
    for key,value in entity_level_dic.items():
        for k,v in value.items():
            if isinstance(v,float) or isinstance(v,int):
                continue
            entity_level_dic[key][k] = v.item()
    for key,value in token_level_dic.items():
        for k,v in value.items():
            if isinstance(v,float) or isinstance(v,int):
                continue
            token_level_dic[key][k] = v.item()
    output["entity_performance"]=entity_level_dic
    output["token_performance"] = token_level_dic
    json_str = json.dumps(output, indent=4)
    with open(data.model_dir+'performance.json', 'w') as json_file:
        json_file.write(json_str)