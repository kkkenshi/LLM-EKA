# -*- coding: utf-8 -*-
# @Time    : 2022/6/1 21:28
# @Author  : Peilin Zhou
# @FileName: data_generator.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import pandas as pd
import os
import random
def negative_sampling(entity_info, sentence, tags,clinical_template_dict, num=2):
    negative_samples = []
    words_list = sentence.split()
    sample_len = entity_info['end_index'] - entity_info['start_index']
    for i in range(num):
        neg_begin = random.randint(0, len(tags) - sample_len)
        while 'O' not in tags[neg_begin:neg_begin+sample_len]:
            neg_begin = random.randint(0, len(tags) - sample_len)
        negative_samples.append([sentence,' '.join(words_list[neg_begin:neg_begin+sample_len])+clinical_template_dict['O']])
    return negative_samples


def tags_to_entities(tags):
    """Note that the end index returned by this function is inclusive.
    To use it for Span creation, increment the end by 1."""
    entities = []
    start = None
    for i, tag in enumerate(tags):
        if tag is None or tag.startswith("-"):
            # TODO: We shouldn't be getting these malformed inputs. Fix this.
            if start is not None:
                start = None
            else:
                entities.append(("", i, i))
        elif tag.startswith("O"):
            pass
        elif tag.startswith("I"):
            if start is None:
                continue

        elif tag.startswith("S"):
            entities.append((tag[2:], i, i))
        elif tag.startswith("B"):
            start = i
        elif tag.startswith("E"):
            if start is None:
#                 print(tags)
                continue
            entities.append((tag[2:], start, i))
            start = None
        else:
            # raise ValueError(Errors.E068.format(tag=tag))
            raise ValueError(tags)
    return entities
def generate_annotations(tokenized_text_list,tags_list):
    annotation_list = []
    for text,tags in zip(tokenized_text_list,tags_list):
        entities = tags_to_entities(tags)
        annotation = []
        for e in entities:
            ret = {}
            start_index = e[1]
            end_index = e[2] + 1
            ret['type'] = e[0]
            ret['start_index'] = start_index
            ret['end_index'] = end_index
            ret['value']=' '.join(text[start_index:end_index])
            annotation.append(ret)
        annotation_list.append(annotation)
    return annotation_list

dataset_name='clinical_ner'
data_type = 'dev'
root_path = '../../datasets/{}/7-1.5-1.5'.format(dataset_name)
path = '{}/{}.bioes'.format(root_path,data_type)
template_list = [" is a location entity .", " is a person entity .", " is an organization entity .",
                 " is an other entity .", " is not a named entity ."]
clinical_template_dict = {'Location':" is a location entity .",
                 'Person':" is a person entity .",
                 'Organization':" is an organization entity .",
                 'Disease':" is a disease entity .",
                 'Symptom':" is a symptom entity .",
                 'Drug':" is a drug entity .",
                 'Vaccine-related':" is a vaccine related entity .",
                 'O':" is not a named entity ."
                 }
wnut16_template_dict = {
        'COMPANY':" is a company entity .",
        'FACILITY':" is a facility entity .",
        'LOC':" is a location entity .",
        'MOVIE':" is a movie entity .",
        'MUSICARTIST':" is a musicartist entity .",
        'OTHER':" is an other entity .",
        'PERSON':" is a person entity .",
        'PRODUCT':" is a product entity .",
        'SPORTSTEAM':" is a sportsteam entity .",
        'TVSHOW':" is a tvshow entity .",
        'O':" is not a named entity ."
}
wnut17_template_dict = {
    'CORPORATION':" is a corporation entity .",
    'GROUP':" is a group entity .",
    'LOCATION':" is a location entity .",
    'PERSON':" is a person entity .",
    'PRODUCT':" is a product entity .",
    'WORK':" is a work entity .",
    'O':" is not a named entity ."
}
template_dict ={
    'clinical_ner':clinical_template_dict,
    'wnut16':wnut16_template_dict,
    'wnut17':wnut17_template_dict
}
with open(path,'r') as f:
    lines = f.readlines()
    lines = [_.strip() for _ in lines]
sentence_list = []
words_list = []
labels_list = []
labels = []
for line in lines:
    if line != '':
        words_list.append(line.split()[0])
        labels.append(line.split()[1])
    else:
        sentence_list.append(' '.join(words_list))
        words_list = []
        labels_list.append(labels)
        labels = []
print(len(sentence_list))
print(sentence_list[0:5])
print(len(labels_list))
triples = []
annotations_list = []
df_dict = {'Source sentence':[],'Answer sentence':[]}
for sentence,labels in zip(sentence_list,labels_list):
    annotations = generate_annotations([sentence.split()],[labels])[0]
    # print(annotations)
    if len(annotations) >=1:
        for entity_info in annotations:
            triples.append([sentence,entity_info['value']+template_dict[dataset_name][entity_info['type']]])
            negative_samples = negative_sampling(entity_info, sentence, labels, clinical_template_dict, num=2)
            triples+=negative_samples
    # for word,label in zip(sentence.split(),labels):
    #     if label=='O':
    #         triples.append([sentence,word+template_dict[dataset_name]['O']])
print(len(triples))
for item in triples:
    df_dict['Source sentence'].append(item[0])
    df_dict['Answer sentence'].append(item[1])
df =pd.DataFrame(df_dict)
df.to_csv('{}/{}.csv'.format(root_path,data_type),index=False,sep='\t')
# with open('a.txt','w') as f:
#     for item in triples:
#         f.writelines(item[0]+'\t'+item[1]+'\n')