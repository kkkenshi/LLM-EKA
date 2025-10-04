# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 20:22
# @Author  : Peilin Zhou
# @FileName: train.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
from yato import YATO
import torch
import argparse
def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    args.add_argument("--dataset_name", default='clinical', type=str)# could be clinical/wnut16/wnut17/conll
    args.add_argument("--config_file", default='decode.config', type=str)

    args.add_argument("--exp_dir",
                      default='./exps/clinical/m_bert-base-cased_s_42/', type=str)
    args.add_argument("--raw_dir", default='../../datasets/clinical_ner/7-1.5-1.5/test.bioes', type=str)#实验存放地址 exps/dataset_name/模型+seed


    ###TrainingSetting###
    args.add_argument("--seed", default=42, type=int)
    args.add_argument("--HP_iteration", default=1, type=int)
    args.add_argument("--HP_batch_size", default=16, type=int)
    ###Hyperparameters###
    args.add_argument("--HP_dropout", default=0.3, type=float)
    args.add_argument("--HP_lr", default=3e-5, type=float)
    args.add_argument( "--HP_gpu", action='store_true')
    args.set_defaults(HP_gpu=True)
    args.add_argument("--device", default=0, type=int)
    args.add_argument("--scheduler", default='get_cosine_schedule_with_warmup', type=str)
    args.add_argument("--warmup_step_rate", default=0.05, type=float)

    args = args.parse_args()

    dataset2path = {
        'clinical_bertweet':('../../datasets/clinical_ner/7-1.5-1.5/bertweet/train.bioes','../../datasets/clinical_ner/7-1.5-1.5/bertweet/dev.bioes','../../datasets/clinical_ner/7-1.5-1.5/bertweet/test.bioes'),
        'clinical':('../../datasets/clinical_ner/7-1.5-1.5/train.bioes','../../datasets/clinical_ner/7-1.5-1.5/dev.bioes','../../datasets/clinical_ner/7-1.5-1.5/test.bioes'),
        'wnut16':('../../datasets/wnut16/train.bioes','../../datasets/wnut16/dev.bioes','../../datasets/wnut16/test.bioes'),
        'wnut17': (
        '../../datasets/wnut17/train.bioes', '../../datasets/wnut17/dev.bioes', '../../datasets/wnut17/test.bioes'),
        'conll':('./datasets/CoNLL03_NER/train.bioes','./datasets/CoNLL03_NER/dev.bioes','./datasets/CoNLL03_NER/test.bioes')
    }
    assert args.dataset_name in ['clinical','clinical_bertweet','wnut16','wnut17','conll']
    args.train_dir,args.dev_dir,args.test_dir = dataset2path[args.dataset_name]
    args.load_model_dir = args.exp_dir +'best_model.pth'
    args.dset_dir = args.exp_dir + 'exp.dset'
    args.model_dir = args.load_model_dir
    args.decode_dir = args.exp_dir + 'prediction.decode'

    params = {}
    for k, v in vars(args).items():
        if v=='':
            params[k] = None
        else:
            params[k] = v
    params['device'] = torch.device('cuda:' + str(args.device))
    return params, args
# 模型训练
if __name__ == '__main__':
    config_dict,args = get_params()
    decode_model = YATO(args.config_file,config_dict=config_dict)#configuration file
    result_dict = decode_model.decode()
