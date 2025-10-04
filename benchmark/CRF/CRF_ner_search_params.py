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
#Modeling
import scipy
import sklearn
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import accuracy_score, classification_report
from sklearn.metrics import  make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn_crfsuite
import os


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
def extract_sentences_from_file(file):
    path = file
    sentences = []
    sentence = []
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [_.strip() for _ in lines]
    for line in lines:
        if line != '':
            sentence.append((line.split()[0], line.split()[1]))
        else:
            sentences.append(sentence)
            sentence = []
    return sentences
def load_data(dataset_name, seed=42):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    dataset2path = {
            'clinical':('../../datasets/clinical_ner/7-1.5-1.5/train.bioes','../../datasets/clinical_ner/7-1.5-1.5/dev.bioes','../../datasets/clinical_ner/7-1.5-1.5/test.bioes'),
            'wnut16':('../../datasets/wnut16/train.bioes','../../datasets/wnut16/dev.bioes','../../datasets/wnut16/test.bioes'),
            'wnut17': (
            '../../datasets/wnut17/train.bioes', '../../datasets/wnut17/dev.bioes', '../../datasets/wnut17/test.bioes'),
            'conll':('./datasets/CoNLL03_NER/train.bioes','./datasets/CoNLL03_NER/dev.bioes','./datasets/CoNLL03_NER/test.bioes')
    }
    train_path, dev_path, test_path = dataset2path[dataset_name]
    train_df = anns_to_df(train_path)
    valid_df = anns_to_df(dev_path)
    test_df = anns_to_df(test_path)
    train_sentences = extract_sentences_from_df(train_df)
    random.shuffle(train_sentences)
    valid_sentences = extract_sentences_from_df(valid_df)
    test_sentences = extract_sentences_from_file(test_path)
    # print(test_sentences[-1])
    # print(len(test_sentences[-1]))
    # print(sum([len(_) for _ in test_sentences]))
    X_train = np.array([sent2features(s) for s in train_sentences])
    y_train = np.array([sent2labels(s) for s in train_sentences])
    X_valid = np.array([sent2features(s) for s in valid_sentences])
    y_valid = np.array([sent2labels(s) for s in valid_sentences])
    X_test = np.array([sent2features(s) for s in test_sentences])
    y_test = np.array([sent2labels(s) for s in test_sentences])
    return X_train,y_train,X_valid,y_valid,X_test,y_test

def search_hyper(X_train, y_train, X_valid, y_valid):
    labels = ['B-Organization', 'E-Organization', 'S-Person', 'S-Location', 'S-Drug', 'S-Organization', 'B-Symptom',
               'E-Symptom', 'S-Symptom', 'I-Symptom', 'I-Organization', 'S-Vaccine-related', 'B-Vaccine-related',
               'E-Vaccine-related', 'S-Disease', 'B-Disease', 'E-Disease', 'B-Location', 'E-Location', 'B-Person',
               'E-Person', 'I-Vaccine-related', 'I-Location', 'I-Disease', 'B-Drug', 'E-Drug', 'I-Person', 'I-Drug']

    # find best hyper-parameters
    ## Using a fixed training-development-test split in sklearn
    ## http://www.wellformedness.com/blog/using-a-fixed-training-development-test-split-in-sklearn/
    x = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    test_fold = np.concatenate([
        # The training data.
        np.full(X_train.shape[0], -1, dtype=np.int8),
        # The development data.
        np.zeros(X_valid.shape[0], dtype=np.int8)
    ])
    cv = sklearn.model_selection.PredefinedSplit(test_fold)
    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        max_iterations=50,
        all_possible_transitions=True
    )
    params_space = {
        'c2': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    }
    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.05),
    # }
    # params_space = {
    #     'c1': [0.0001,0.001,0.01,0.1,1,10],
    #     'c2': [0.0001,0.001,0.01,0.1,1,10],
    # }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = GridSearchCV(crf, params_space,
                      cv=cv,
                      verbose=1,
                      n_jobs=-1,
                      scoring=f1_scorer)
    # rs = RandomizedSearchCV(crf, params_space,
    #                         cv=cv,
    #                         verbose=1,
    #                         n_jobs=-1,
    #                         n_iter=50,
    #                         scoring=f1_scorer)
    rs.fit(x, y)
    # crf = rs.best_estimator_
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    return rs.best_params_
def save_prediction( prediction, output_path):
    with open('../../datasets/clinical_ner/7-1.5-1.5/test.bioes','r') as f:
        lines  = f.readlines()
        lines = [_.strip() for _ in lines]
    sentences = []
    sentence = []
    for line in lines:
        if line!='':
            word = line.split()[0]
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence = []
    output_file_path = output_path + 'test_prediction.txt'
    with open(output_file_path,'w') as f:
        for i,(sentence,tags) in enumerate(zip(sentences,prediction)):
            for word,tag in zip(sentence,tags):
                f.writelines('{} {}\n'.format(word,tag))
            f.writelines('\n')

def train_and_test(model,dataset_name,seed,output_path):
    X_train, y_train, _, _, X_test, y_test = load_data(dataset_name=dataset_name, seed=seed)
    ## 模型训练
    model.fit(X_train, y_train)
    ## 模型评估
    y_pred = model.predict(X_test)
    print(len(X_test[-1]))

    save_prediction(y_pred, output_path=output_path)
    labels = list(model.classes_)
    labels.remove('O')
    report_f1(y_test, y_pred, output_path=output_path)


if __name__ == '__main__':
    # dataset_name = 'clinical'
    # dataset_name_list = ['clinical','wnut16','wnut17']
    dataset_name_list = ['clinical']
    seed_list = [22, 32, 42, 52, 62]
    # seed_list = [22]
    for dataset_name in dataset_name_list:
        print('正在处理数据集:{}'.format(dataset_name))
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(dataset_name=dataset_name,seed=42)

        best_hyper = search_hyper(X_train, y_train, X_valid, y_valid)
        crf = sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=best_hyper['c2'],
            max_iterations=50,
            all_possible_transitions=True
        )
        for seed in seed_list:
            output_path = './m_crf_d_{}_s_{}/'.format(dataset_name,seed)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            train_and_test(crf,dataset_name=dataset_name,seed=seed,output_path=output_path)
        print('数据集:{}已处理！！'.format(dataset_name))

