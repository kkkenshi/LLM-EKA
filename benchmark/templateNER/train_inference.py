from utils_metrics import get_entities_bio, f1_score
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import time
import math
import pandas as pd
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import accuracy_score, classification_report
import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
import argparse

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    args.add_argument("--dataset_name", default='wnut16', type=str)  # could be clinical/wnut16/wnut17/conll
    args.add_argument("--output_dir", default='./exp/', type=str)  # 实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--encoder_decoder_name", default="../../pretrained_models/bart/bart-base",
                      type=str)  # 实验存放地址 exps/dataset_name/模型+seed

    args.add_argument("--num_train_epochs", default=1, type=int)
    args.add_argument("--train_batch_size", default=70, type=int)
    args.add_argument("--max_seq_length", default=50, type=int)
    args.add_argument("--max_length", default=25, type=int)
    args.add_argument("--device", default=0, type=int)
    args.add_argument("--seed", default=42, type=int)

    args = args.parse_args()
    args.output_dir = './exp/m_{}_d_{}_s_{}'.format(args.encoder_decoder_name.split('/')[-1],args.dataset_name,args.seed)

    dataset2path = {
        'clinical': ('../../datasets/clinical_ner/7-1.5-1.5/train.csv', '../../datasets/clinical_ner/7-1.5-1.5/dev.csv',
                     '../../datasets/clinical_ner/7-1.5-1.5/test.bioes'),
        'wnut16': (
        '../../datasets/wnut16/train.csv', '../../datasets/wnut16/dev.csv', '../../datasets/wnut16/test.bioes'),
        'wnut17': (
            '../../datasets/wnut17/train.csv', '../../datasets/wnut17/dev.csv', '../../datasets/wnut17/test.bioes'),
        'conll': (
        './datasets/CoNLL03_NER/train.csv', './datasets/CoNLL03_NER/dev.csv', './datasets/CoNLL03_NER/test.bioes')
    }
    assert args.dataset_name in ['clinical', 'wnut16', 'wnut17', 'conll']
    args.train_dir, args.dev_dir, args.test_dir = dataset2path[args.dataset_name]

    params = {}
    for k, v in vars(args).items():
        if v == '':
            params[k] = None
        else:
            params[k] = v
    return params, args

def report_f1(golden_list,predict_list,output_path=None):
    labels = set()
    for items in golden_list:
        labels = labels | set(items)
    labels = list(labels)
    labels.remove('O')

    print(classification_report(golden_list, predict_list, scheme='IOBES', digits=4))
    print(flat_classification_report(golden_list, predict_list, labels = labels, digits=4))
    report_entity_level = classification_report(golden_list, predict_list, scheme='IOBES', digits=4, output_dict=True)
    report_token_level = flat_classification_report(golden_list, predict_list, labels = labels, digits=4, output_dict=True)
    if output_path:
        df = pd.DataFrame(report_entity_level).transpose()
        df.to_csv(output_path+'entity_level_eval.csv', index=True)
        df = pd.DataFrame(report_token_level).transpose()
        df.to_csv(output_path+'token_level_eval.csv', index=True)
    return report_entity_level,report_token_level
class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def template_entity(words, input_TXT, start, dataset='clinical'):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    if dataset == 'clinical':
        template_list = [" is a location entity .",
                         " is a person entity .",
                         " is an organization entity .",
                         " is a disease entity .",
                         " is a symptom entity .",
                         " is a drug entity .",
                         " is a vaccine related entity .",
                         " is not a named entity ."]
        entity_dict = {0: 'Location', 1: 'Person', 2: 'Organization', 3: 'Disease', 4: 'Symptom', 5: 'Drug',
                       6: 'Vaccine-related', 7: 'O'}
    elif dataset == 'wnut16':
        template_list = [" is a company entity .",
                         " is a facility entity .",
                         " is a location entity .",
                         " is a movie entity .",
                         " is a musicartist entity .",
                         " is an other entity .",
                         " is a person entity .",
                         " is a product entity .",
                         " is a sportsteam entity .",
                         " is a tvshow entity .",
                         " is not a named entity ."]
        entity_dict = {0: 'COMPANY', 1: 'FACILITY', 2: 'LOC', 3: 'MOVIE', 4: 'MUSICARTIST', 5: 'OTHER',
                       6: 'PERSON',7:'PRODUCT', 8:'SPORTSTEAM',9:'TVSHOW',10: 'O'}
    elif dataset == 'wnut17':
        template_list = [" is a corporation entity .",
                         " is a group entity .",
                         " is a location entity .",
                         " is a person entity .",
                         " is a product entity .",
                         " is a work entity .",
                         " is not a named entity ."]
        entity_dict = {0: 'CORPORATION', 1: 'GROUP', 2: 'LOCATION', 3: 'PERSON', 4: 'PRODUCT', 5: 'WORK',
                       6: 'O'}
    tag_type_num = len(entity_dict)
    input_TXT = [input_TXT]*(tag_type_num*words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)

    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0]*tag_type_num*words_length


    for i in range(len(temp_list)//tag_type_num):
        base_length = ((tokenizer(temp_list[i * tag_type_num], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        #1) base_length = ((tokenizer(temp_list[i * tag_type_num], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - (tag_type_num-1)
        output_length_list[i*tag_type_num:i*tag_type_num+ tag_type_num] = [base_length]*tag_type_num
        output_length_list[i*tag_type_num+tag_type_num-1] += 1
        #2) output_length_list[i * tag_type_num + 4] += 1

    score = [1]*tag_type_num*words_length
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]
        #3) for i in range(output_ids.shape[1] - (tag_type_num-2)):
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, tag_type_num*words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start+(score.index(max(score))//tag_type_num)
        # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score))%tag_type_num)], max(score)] #[start_index,end_index,label,score]



def prediction(input_TXT,dataset='clinical'):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append(word)

        entity = template_entity(words, input_TXT, i,dataset=dataset) #[start_index,end_index,label,score]
        if entity[1] >= len(input_TXT_list):
            entity[1] = len(input_TXT_list)-1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)

    for entity in entity_list:
        dis = entity[1] -entity[0]
        if dis==0:
            label_list[entity[0]:entity[1] + 1] = ["S-" + entity[2]] * (entity[1] - entity[0] + 1)
        elif dis==1:
            label_list[entity[0]] = "B-" + entity[2]
            label_list[entity[1]] = "E-" + entity[2]
        else:
            label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
            label_list[entity[0]] = "B-"+entity[2]
            label_list[entity[1]] = "E-" + entity[2]
    return label_list

def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



if __name__ == '__main__':
    config_dict, args = get_params()
    # train_data = pd.read_csv(args.train_dir, sep='\t').values.tolist()
    # train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
    #
    # eval_data = pd.read_csv(args.dev_dir, sep='\t').values.tolist()
    # eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
    train_data = pd.read_csv(args.train_dir, sep='\t').values.tolist()
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    eval_data = pd.read_csv(args.dev_dir, sep='\t').values.tolist()
    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": True,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": args.max_length,
        "manual_seed": args.seed,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": args.output_dir,
        "best_model_dir": args.output_dir + '/best_model'
    }
    print(model_args)
    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=args.encoder_decoder_name,
        args=model_args,
        # use_cuda=False,
        cuda_device=args.device
    )

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    results = model.eval_model(eval_df)

    tokenizer = BartTokenizer.from_pretrained(model_args['best_model_dir'])
    model = BartForConditionalGeneration.from_pretrained(model_args['best_model_dir'])
    model.eval()
    model.config.use_cache = False
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    score_list = []
    file_path = args.test_dir
    guid_index = 1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))

    trues_list = []
    preds_list = []
    str = ' '
    num_01 = len(examples)
    num_point = 0
    start = time.time()

    for example in examples:
        sources = str.join(example.words)
        preds_list.append(prediction(sources,dataset=args.dataset_name))
        trues_list.append(example.labels)
        print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
        print('Pred:', preds_list[num_point])
        print('Gold:', trues_list[num_point])
        num_point += 1
    report_f1(trues_list,preds_list,output_path=model_args['best_model_dir']+'/')
    for num_point in range(len(preds_list)):
        preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
        trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
    with open(model_args['best_model_dir']+'/pred.txt', 'w') as f0:
        f0.writelines(preds_list)
    with open(model_args['best_model_dir']+'/gold.txt', 'w') as f0:
        f0.writelines(trues_list)
