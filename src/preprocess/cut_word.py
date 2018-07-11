# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 上午11:26
# @Author  : zhanzecheng
# @File    : cut_word.py
# @Software: PyCharm
"""
'''
我们使用jieba来分词
'''
import jieba
import tqdm
import os
import sys
if __name__ == '__main__':
    FILENAME = os.path.join('../../data/', sys.argv[1])
    SAVENAME = os.path.join('../../data/', sys.argv[1].replace('_to_', '_cut_'))
    STOPWORD_FILE = '../../data/stopword.txt'
    ORIGIN_LABEL_FILE = '../../data/News_pic_label_train.txt'

    # 加载原始label文件
    origin_label = {}
    with open(ORIGIN_LABEL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            origin_label[line[0]] = line[1]

    print('------------> begin to cut words')
    with open(FILENAME, 'r', encoding='utf-8') as f, open(SAVENAME, 'w', encoding='utf-8') as d:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line = line.strip()
            line = line.split('\t')
            result = line[0]
            for l in line[1:]:
                result += '\t'
                seg_list = jieba.cut(l, cut_all=False)
                for count, seg in enumerate(seg_list):
                    if count == 0:
                        result += seg
                    else:
                        result = result + ' ' + seg
            d.write(result + '\n')

    print('------------> cut words done')



    ORIGIN_VALID_FILE = SAVENAME
    VALID_FILE = ORIGIN_VALID_FILE.replace('News', 'All')

    STOPWORD = []
    # 加载stopword
    with open(STOPWORD_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            STOPWORD.append(line)

    # 加载原文件 valid
    with open(ORIGIN_VALID_FILE, 'r', encoding='utf-8') as f, open(VALID_FILE, 'w', encoding='utf-8') as d:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line = line.strip()
            line = line.split('\t')
            id = line[0]
            text = line[1:]
            result = ""
            for te in text:
                te = te.strip().split(' ')
                for t in te:
                    if t not in STOPWORD:
                        result = result + ' ' + t
            if id in origin_label:
                label = origin_label[id]
                d.write(id + '\t' + result + '\t' + label + '\n')
            else:
                d.write(id + '\t' + result + '\n')


