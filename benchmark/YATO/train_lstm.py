from yato import YATO
import torch
import argparse
def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    args.add_argument("--dataset_name", default='clinical', type=str)# could be clinical/wnut16/wnut17/conll
    args.add_argument("--model_dir", default='sample_data/', type=str)#实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--dset_dir", default='exp.dset', type=str)#实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--config_file", default='./config_lists/covid.bilstm_crf.config', type=str)

    args.add_argument("--word_emb_dim", default=50, type=int)
    args.add_argument("--char_emb_dim", default=30, type=int)

    args.add_argument("--norm_word_emb", action='store_true')
    args.add_argument("--norm_char_emb", action='store_true')
    args.add_argument("--ave_batch_loss", action='store_true')

    ###NetworkConfiguration###
    args.add_argument("--use_crf", action='store_true')
    # args.set_defaults(use_crf=True)
    args.add_argument( "--use_char", action='store_true')
    # args.set_defaults(use_char=True)
    args.add_argument( "--char_feature_extractor", default='None', type=str)

    args.add_argument( "--use_word_seq", action='store_true')
    args.set_defaults(use_word_seq=True)
    args.add_argument( "--use_word_emb", action='store_true')
    args.set_defaults(use_word_emb=True)
    args.add_argument( "--word_feature_extractor", default='LSTM', type=str)


    ###TrainingSetting###
    args.add_argument("--seed", default=42, type=int)
    args.add_argument("--HP_iteration", default=50, type=int)
    args.add_argument("--HP_batch_size", default=16, type=int)
    args.add_argument("--status", default='train', type=str)
    args.add_argument("--optimizer", default='SGD', type=str)

    ###Hyperparameters###
    args.add_argument("--word_cutoff", default=0, type=int)
    args.add_argument("--HP_cnn_layer", default=4, type=int)
    args.add_argument("--HP_char_hidden_dim", default=50, type=int)
    args.add_argument("--HP_hidden_dim", default=200, type=int)
    args.add_argument("--HP_dropout", default=0.5, type=float)
    args.add_argument("--HP_lstm_layer", default=1, type=int)
    args.add_argument( "--HP_bilstm", action='store_true')
    args.set_defaults(HP_bilstm=True)
    args.add_argument("--HP_lr", default=0.015, type=float)
    args.add_argument("--lr_decay", default=0.05, type=float)
    args.add_argument("--HP_momentum", default=0, type=int)
    args.add_argument("--HP_l2", default=1e-8, type=float)
    args.add_argument( "--HP_gpu", action='store_true')
    args.set_defaults(HP_gpu=True)
    args.add_argument("--device", default=0, type=int)
    args.add_argument("--scheduler", default='None', type=str)
    args.add_argument("--warmup_step_rate", default=0.05, type=float)

    args = args.parse_args()
    dataset2path = {
        'clinical':('../../datasets/clinical_ner/7-1.5-1.5/train.bioes','../../datasets/clinical_ner/7-1.5-1.5/dev.bioes','../../datasets/clinical_ner/7-1.5-1.5/test.bioes'),
        'wnut16':('../../datasets/wnut16/train.bioes','../../datasets/wnut16/dev.bioes','../../datasets/wnut16/test.bioes'),
        'wnut17': (
        '../../datasets/wnut17/train.bioes', '../../datasets/wnut17/dev.bioes', '../../datasets/wnut17/test.bioes'),
        'conll':('./datasets/CoNLL03_NER/train.bioes','./datasets/CoNLL03_NER/dev.bioes','./datasets/CoNLL03_NER/test.bioes')
    }
    assert args.dataset_name in ['clinical','wnut16','wnut17','conll']
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
# 模型预测
# decode_model = YATO('decode.config')#configuration file
# result_dict = decode_model.decode()
# print('hi')

