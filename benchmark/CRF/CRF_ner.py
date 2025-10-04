# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 19:37
# @Author  : Peilin Zhou
# @FileName: CRF_ner.py.py
# @Software: PyCharm
# @E-mail  : zhoupl@pku.edu.cn
# -*- coding: utf-8 -*-
## 探索不同特征组合对性能的影响
#Data analysis
import pandas as pd
import numpy as np
import random
#Data visualisation
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=True)
# sns.set(font_scale=1)
#Modeling
import sklearn
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import accuracy_score, classification_report
from sklearn.metrics import  make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
import scipy.stats
# import eli5
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
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),   #小写
        # 'word[0]': word[0],         #首字母
        # 'word[0:2]': word[0:2],         #前两个字母
        # 'word[0:3]': word[0:3],         #前三个字母
        'word[-3:]': word[-3:],         #后三个字符
        'word[-2:]': word[-2:],         #后两个字符
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),#检查首字符是否大些，且其他字母为小写
        'word.isdigit()': word.isdigit(),
        # 'word.@': True if word[0]=='@' else False,
        # 'word.#': True if word[0]=='#' else False
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            # '-1:word.isdigit()': word1.isdigit(),
            # '-1:word[-3:]': word1[-3:],         #后三个字符
            # '-1:word[-2:]': word1[-2:],         #后两个字符
        })
        # if i>1:
        #     word2 = sent[i-2][0]
        #     features.update({
        #         '-2:word.lower()': word2.lower(),
        #         '-2:word.istitle()': word2.istitle(),
        #         '-2:word.isupper()': word2.isupper(),
        #         # '-1:word.isdigit()': word1.isdigit(),
        #         # '-1:word[-3:]': word1[-3:],         #后三个字符
        #         # '-1:word[-2:]': word1[-2:],         #后两个字符
        #     })

    else:
        #如果是句子中的第一个单词，额外增加一个特征BOS
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            # '+1:word.isdigit()': word1.isdigit(),
            # '+1:word[-3:]': word1[-3:],         #后三个字符
            # '+1:word[-2:]': word1[-2:],         #后两个字符
        })
        # if i < len(sent)-2:
        #     word2 = sent[i+2][0]
        #     features.update({
        #         '+2:word.lower()': word2.lower(),
        #         '+2:word.istitle()': word2.istitle(),
        #         '+2:word.isupper()': word2.isupper(),
        #         # '+1:word.isdigit()': word1.isdigit(),
        #         # '+1:word[-3:]': word1[-3:],         #后三个字符
        #         # '+1:word[-2:]': word1[-2:],         #后两个字符
        #     })
    else:
        #如果是句子中的最后一个单词，额外增加一个特征EOS
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, label in sent]
def anns_to_df(path):
    count=1
    word=[]
    tag=[]
    sentence = []
    s = []
    s_id = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            if line!='':
                word.append(line.split()[0])
                s.append(line.split()[0])
                tag.append(line.split()[1])
                s_id.append('Sentence: {}'.format(count))
            else:
                count+=1
                sentence.append(' '.join(s))
                s= []
    dic = {'Sentence #':s_id,'Word':word,'Tag':tag}
    data=pd.DataFrame(dic)#将字典转换成为数据框
    return data
def extract_sentences_from_df(data):
    agg_func = lambda s: [(w, t) for w, t in zip(s['Word'].values.tolist(),
                                                    s['Tag'].values.tolist())]
    grouped_df = data.groupby('Sentence #').apply(agg_func)
    sentences = [s for s in grouped_df]
    return sentences
SEED = 42
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.
train_df = anns_to_df('../../datasets/clinical_ner/7-1.5-1.5/train.bioes')
valid_df = anns_to_df('../../datasets/clinical_ner/7-1.5-1.5/dev.bioes')
test_df = anns_to_df('../../datasets/clinical_ner/7-1.5-1.5/test.bioes')
cross_df = anns_to_df('../../datasets/clinical_ner/7-1.5-1.5/cross-8500.anns')
train_sentences = extract_sentences_from_df(train_df)
random.shuffle(train_sentences)
valid_sentences = extract_sentences_from_df(valid_df)
test_sentences = extract_sentences_from_df(test_df)
cross_sentences = extract_sentences_from_df(cross_df)
X_train = np.array([sent2features(s) for s in train_sentences])
y_train = np.array([sent2labels(s) for s in train_sentences])
X_valid = np.array([sent2features(s) for s in valid_sentences])
y_valid = np.array([sent2labels(s) for s in valid_sentences])
X_test = np.array([sent2features(s) for s in test_sentences])
y_test = np.array([sent2labels(s) for s in test_sentences])
X_cross = np.array([sent2features(s) for s in cross_sentences])
y_cross = np.array([sent2labels(s) for s in cross_sentences])

# 训练CRF
crf = sklearn_crfsuite.CRF(algorithm='l2sgd',
                           # c1= 0.01,
                           c2=0.01,
                           max_iterations=50,
                           all_possible_transitions=False,
                           verbose=True)
## 模型训练
crf.fit(X_train, y_train)
# crf.fit(X_valid, y_valid)
# crf.fit(X_test, y_test)
## 模型评估
y_pred = crf.predict(X_test)
# y_pred = crf.predict(X_train)
labels = list(crf.classes_)
labels.remove('O')
report_f1(y_test, y_pred, output_path='./')
