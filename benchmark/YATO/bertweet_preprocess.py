# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 02:44
# @Author  : Peilin Zhou
# @FileName: bertweet_preprocess.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
import pandas as pd
from transformers import AutoTokenizer
def cut_sentence_tags(sentence,tags,tokenizer,max_len = 128):
    word_tokens_len = 0
    new_sentence = []
    new_tags = []
    for word,tag in zip(sentence,tags):
        new_sentence.append(word)
        new_tags.append(tag)
        word_tokens_len+=len(tokenizer.tokenize(word))
        if word_tokens_len > max_len:
            new_sentence = new_sentence[:-1]
            new_tags = new_tags[:-1]
            if new_tags[-1]!='O':
                raise ValueError
            break
    return new_sentence,new_tags

path = '../../datasets/clinical_ner/7-1.5-1.5/'
train_path = path + 'train.bioes'
dev_path = path + 'dev.bioes'
test_path = path + 'test.bioes'
tokenizer = AutoTokenizer.from_pretrained('../../pretrained_models/twitterPLM/bertweet-covid19-base-cased')
bioes_path = test_path
with open(bioes_path,'r') as f:
    lines = f.readlines()
    lines = [_.strip() for _ in lines]
train_samples = []
train_tags = []
temp_sample = []
temp_tags = []
temp_len = []
train_tokenized_len = []
for line in lines:
    if line != '':
        word = line.split()[0]
        tag = line.split()[1]
        tokenized_result = tokenizer.tokenize(word)
        temp_len.append(len(tokenized_result))
        temp_sample.append(word)
        temp_tags.append(tag)
    else:
        train_samples.append(temp_sample)
        train_tags.append(temp_tags)
        train_tokenized_len.append(temp_len)
        temp_sample = []
        temp_tags = []
        temp_len = []
print(len(train_samples))
print(train_samples[0])
print(tokenizer.tokenize('how are you'))
print(train_tokenized_len[0])
count = 0
count_ = 0
processed_sentences = []
processed_tags =[]
for sentence,tags,length in zip(train_samples,train_tags,train_tokenized_len):
    new_sentence,new_tags = cut_sentence_tags(sentence,tags,tokenizer,120)
    if len(new_sentence)!=len(sentence):
        count+=1
    processed_sentences.append(new_sentence)
    processed_tags.append(new_tags)
print(len(processed_sentences))
print(count)
with open( '../../datasets/clinical_ner/7-1.5-1.5/bertweet/test.bioes','w') as f:
    for sentence,tags in zip(processed_sentences,processed_tags):
        for word,tag in zip(sentence,tags):
            f.writelines(word+' '+tag+'\n')
        f.writelines('\n')
# for sentence,tag,length in zip(train_samples,train_tags,train_tokenized_len):
#     if sum(length)>128:
#         print(sentence)
#         # print(length)
#         # print(len(sentence))
#         for i in range(len(tag)-1,-1,-1):
#             if tag[i]!='O':
#                 print(sum(length[:i+1]))
#                 break
#         for index,t in enumerate(tag):
#             if index>128 and t!='O':
#                 count_ +=1
#                 break
#         count+=1
# print(count)
# print(count_)
# for sentence,tag,length in zip(train_samples,train_tags,train_tokenized_len):
#     if length>128:
#         print(sentence)
#         print(length)
#         for index,t in enumerate(tag):
#             if index>128 and t!='O':
#                 count_ +=1
#                 break
#         count+=1

