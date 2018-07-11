# -*- coding: utf-8 -*-
"""
# @Time    : 2018/4/6 下午11:08
# @Author  : zhanzecheng
# @File    : process_features.py
# @Software: PyCharm
"""

import pickle
import pandas as pd
import tqdm
import os
import sys
if __name__ == '__main__':
    if sys.argv[1] == 'test':
        name = 'test'
        FEATURES_test_FILE  = os.path.join('../../data/', 'News_to_test_text.csv')
        TEST_FILE = os.path.join('../../data/', 'All_cut_test_text.txt')
    elif sys.argv[1] == 'train':
        name = 'train'
        FEATURES_test_FILE = os.path.join('../../data/',  'News_to_train_text.csv')
        TEST_FILE = os.path.join('../../data/', 'All_cut_train_text.txt')
    elif sys.argv[1] == 'ocr_train':
        name = 'ocr_train'
        FEATURES_test_FILE = os.path.join('../../data/', 'News_ocr_train_to_text.csv')
        TEST_FILE = os.path.join('../../data/', 'All_ocr_train_cut_text.txt')
    else:
        name = 'ocr_test'
        FEATURES_test_FILE = os.path.join('../../data/', 'News_ocr_test_to_text.csv')
        TEST_FILE = os.path.join('../../data/', 'All_ocr_test_cut_text.txt')

    ITEM_TO_ID = '../../data/item_to_id_small.pkl'
    STOP_WORD_FILE = '../../data/stopword.txt'

    #####################################
    test_x = []

    with open(ITEM_TO_ID, 'rb') as f:
        item_to_id = pickle.load(f)


    df_test = pd.read_csv(FEATURES_test_FILE)

    STOP_WORD = set()
    UNK = len(item_to_id)
    word_count = {}

    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            STOP_WORD.add(line)

    test_ids = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            test_ids.append(line[0])
            line = line[1]
            test_x.append(line)

    test_features = []
    print('--------> begin save test_features')
    for id in tqdm.tqdm(test_ids):
        tmp = list(df_test.loc[df_test.id == id].values[0])
        assert tmp[6][0] == 'D', print('something wrong during del features')
        del tmp[6]
        test_features.append(tmp)

    assert len(test_features) == len(test_ids)
    with open(os.path.join('../../data/',name + '_features.pkl'), 'wb') as f:
        pickle.dump(test_features, f)

    # 现在来提取TF-IDF的特征

    print('--------> begin load pkl' )
    with open('../../data/make_pipeline.pkl', 'rb') as f:
        lsa = pickle.load(f)
    print('--------> begin transform test_x' )
    test_x = lsa.transform(test_x)

    if name != 'train':
        with open(os.path.join('../../data/', name + '_x_250.pkl'), 'wb') as f:
            pickle.dump(test_x, f)
