from utils_metrics import get_entities_bio, f1_score
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import time
import math
import pandas as pd
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import accuracy_score, classification_report
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

def template_entity(words, input_TXT, start):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT]*(8*words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [" is a location entity .",
                     " is a person entity .",
                     " is an organization entity .",
                     " is a disease entity .",
                     " is a symptom entity .",
                     " is a drug entity .",
                     " is a vaccine related entity .",
                     " is not a named entity ."]
    entity_dict = {0: 'Location', 1: 'Person', 2: 'Organization', 3: 'Disease', 4: 'Symptom',5:'Drug',6:'Vaccine-related',7:'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0]*8*words_length


    for i in range(len(temp_list)//8):
        base_length = ((tokenizer(temp_list[i * 8], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*8:i*8+ 8] = [base_length]*8
        output_length_list[i*8+7] += 1

    score = [1]*8*words_length
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 8*words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start+(score.index(max(score))//8)
        # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score))%8)], max(score)] #[start_index,end_index,label,score]



def prediction(input_TXT):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append(word)

        entity = template_entity(words, input_TXT, i) #[start_index,end_index,label,score]
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


tokenizer = BartTokenizer.from_pretrained('./outputs/best_model')
# input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."
model = BartForConditionalGeneration.from_pretrained('./outputs/best_model')
# model = BartForConditionalGeneration.from_pretrained('../dialogue/bart-large')
model.eval()
model.config.use_cache = False
# input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
# print(input_ids)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

score_list = []
file_path = './data/clinical_ner/test.bioes'
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
    preds_list.append(prediction(sources))
    trues_list.append(example.labels)
    print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
    print('Pred:', preds_list[num_point])
    print('Gold:', trues_list[num_point])
    num_point += 1

report_f1(trues_list,preds_list)
# true_entities = get_entities_bio(trues_list)
# pred_entities = get_entities_bio(preds_list)
# results = {
#     "f1": f1_score(true_entities, pred_entities)
# }
# print(results["f1"])
for num_point in range(len(preds_list)):
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
with open('./pred.txt', 'w') as f0:
    f0.writelines(preds_list)
with open('./gold.txt', 'w') as f0:
    f0.writelines(trues_list)
