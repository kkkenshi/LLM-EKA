from yato import YATO
import torch
import argparse
def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    args.add_argument("--dataset_name", default='clinical_bertweet', type=str)# could be clinical/wnut16/wnut17/conll
    args.add_argument("--model_dir", default='sample_data/', type=str)#实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--dset_dir", default='exp.dset', type=str)#实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--config_file", default='./config_lists/demo.bert.config', type=str)

    args.add_argument("--high_level_transformer", default='../../pretrained_models/twitterPLM/bertweet-covid19-base-cased', type=str)
    # args.add_argument("--high_level_transformer", default='../../pretrained_models/roberta/roberta-base', type=str)
    args.add_argument("--high_level_transformer_finetune", action='store_true')
    args.set_defaults(high_level_transformer_finetune=True)

    args.add_argument("--use_crf", action='store_true')
    # args.set_defaults(use_crf=True)
    
    ###TrainingSetting###
    args.add_argument("--seed", default=42, type=int)
    args.add_argument("--HP_iteration", default=1, type=int)
    args.add_argument("--HP_batch_size", default=12, type=int)
    args.add_argument("--status", default='train', type=str)
    args.add_argument("--optimizer", default='AdamW', type=str)

    ###Hyperparameters###
    args.add_argument("--HP_hidden_dim", default=768, type=int)
    args.add_argument("--HP_dropout", default=0.3, type=float)
    args.add_argument("--HP_lr", default=3e-5, type=float)
    args.add_argument( "--HP_gpu", action='store_true')
    args.set_defaults(HP_gpu=True)
    args.add_argument("--device", default=0, type=int)
    args.add_argument("--scheduler", default='get_cosine_schedule_with_warmup', type=str)
    args.add_argument("--warmup_step_rate", default=0.05, type=float)

    args = args.parse_args()

    dataset2path = {
        'clinical':('/gemini/code/COVID-CETS-master/datasets/output_50-shot.bio','/gemini/data-3/dev.bio','/gemini/data-3/test.bio'),
        'BIORED':('/gemini/data-3/RedBio-5-2-shot.bio','/gemini/data-3/dev.bio','/gemini/data-3/test.bio')
    }
    assert args.dataset_name in ['clinical','BIORED']
    args.train_dir,args.dev_dir,args.test_dir = dataset2path[args.dataset_name]

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
    model = YATO(args.config_file,config_dict=config_dict)#configuration file
    model.train(log='train.log', metric='F')
