# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 20:01
# @Author  : Peilin Zhou
# @FileName: results_collection.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import pandas as pd
import json
import glob
import os
def read_json(params):
    with open(params,'r') as f:
        params_dict = json.load(f)
    return params_dict
def generate_each_dic_lstm(params_file, performance_file):
    model_name = params_file.split('/')[-2].split('_s_')[0][2:]
    entity_performance = read_json(performance_file)['entity_performance']
    flat_entity_performance = {}
    for key,value in entity_performance.items():
        for metric,score in value.items():
            if metric!='support':
                flat_entity_performance[key+'_'+metric] = score
    all_parameters = read_json(params_file)['params']
    key_params = {'model':model_name,
                'dataset': all_parameters['dataset_name'],
                  'epochs': all_parameters['HP_iteration'],
                  'seed': all_parameters['seed'],
                  'lr': all_parameters['HP_lr'],
                  'batch_size': all_parameters['HP_batch_size']
                }
    key_params.update(flat_entity_performance)
    return  key_params
def generate_each_dic(params_file, performance_file):
    entity_performance = read_json(performance_file)['entity_performance']
    flat_entity_performance = {}
    for key,value in entity_performance.items():
        for metric,score in value.items():
            if metric!='support':
                flat_entity_performance[key+'_'+metric] = score
    all_parameters = read_json(params_file)['params']
    key_params = {'model':all_parameters['high_level_transformer'].split('/')[-1],
                'dataset': all_parameters['dataset_name'],
                  'epochs': all_parameters['HP_iteration'],
                  'seed': all_parameters['seed'],
                  'lr': all_parameters['HP_lr'],
                  'batch_size': all_parameters['HP_batch_size']
                }
    key_params.update(flat_entity_performance)
    return  key_params

def generate_df(dataset):
    exps_path_list = glob.glob('./exp/{}/m*'.format(dataset))
    key_params_list = []
    for exps_path in exps_path_list:
        params_file = exps_path+'/params.json'
        performance_file = exps_path+'/performance.json'
        if os.path.exists(params_file) and os.path.exists(performance_file):
            if 'lstm' in params_file:
                key_params = generate_each_dic_lstm(params_file,performance_file)
            else:
                key_params = generate_each_dic(params_file,performance_file)
            key_params_list.append(key_params)
    results_dic = {}
    for key in key_params_list[0].keys():
        if key not in results_dic:
            results_dic[key] = []
        for item in key_params_list:
            results_dic[key].append(item[key])
    df = pd.DataFrame(results_dic)
    return df

# dataset_list = ['clinical','wnut16','wnut17']
dataset_list = ['clinical']
writer = pd.ExcelWriter('all.xlsx')  # 重点1：writer不能在下面的for循环中
counter = 0
for dataset in dataset_list:
    df = generate_df(dataset)
    df.to_excel(writer, dataset)
writer.save()  # 重点2：save不能在for循环里
writer.close()
print('实验结果已保存至results.xlsx!')
# print(results_dic)