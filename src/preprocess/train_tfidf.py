# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/8 上午9:52
# @Author  : zhanzecheng
# @File    : train_tfidf.py
# @Software: PyCharm
"""



import pickle
from sklearn.feature_extraction.text import (
    TfidfVectorizer)
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline


TRAIN_X = '../../data/All_cut_train_text.txt'
STOP_WORD_FILE = '../../data/stopword.txt'
#####################################
train_x = []
ids = []
STOP_WORD = set()

with open(STOP_WORD_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        STOP_WORD.add(line)

with open(TRAIN_X, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        id = line[0]
        ids.append(id)
        label = int(line[2])
        line = line[1]
        train_x.append(line)



# vectorizer text
vectorizer = TfidfVectorizer(ngram_range=(1,2),
                             stop_words=STOP_WORD,
                             sublinear_tf=True,
                             use_idf=True,
                             norm='l2',
                             max_features=10000)
# LSA Pipeline
svd = TruncatedSVD(n_components=250)
lsa = make_pipeline(vectorizer, svd)
# fit lsa
print('begin')
lsa.fit(train_x)
# print('end_fit')
print('--------> begin load pkl' )
with open('../../data/make_pipeline.pkl', 'wb') as f:
    pickle.dump(lsa, f)
print('--------> begin transform train_x' )
train_x = lsa.transform(train_x)
print('--------> begin transform test_x' )
with open('../../data/train_x_250.pkl', 'wb') as f:
    pickle.dump(train_x, f)