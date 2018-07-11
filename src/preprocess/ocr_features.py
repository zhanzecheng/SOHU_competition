# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/10 上午9:47
# @Author  : zhanzecheng
# @File    : ocr_features.py
# @Software: PyCharm
"""
import pickle
import sys
if __name__ == '__main__':
    if sys.argv[1] == 'test':
        name = 'test'
    else:
        name = 'train'
    data = {}
    with open('../../data/result_ocr.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            ID = line.split('\t')[0].split('.')[0]
            tmp = ""
            line = line.split('\t')[1:]
            for l in line:
                tmp += l
            data[ID] = tmp

    IDs = []
    with open('../../data/All_cut_' + name +  '_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')[0]
            IDs.append(line)

    new2img = {}
    # 我们把数据拼接起来, 并分好词
    with open('../../data/News_info_' + name + '.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            id = line.split('\t')[0]
            line = line.split('\t')[-1].split(';')
            tmp = ""
            for l in line:
                l = l.split('.')[0]
                if l in data:
                    tmp = tmp + data[l]
            new2img[id] = tmp

    with open('../../data/News_ocr_' + name + '_to_text.txt', 'w', encoding='utf-8') as f:
        for id in IDs:
            f.write(id + '\t' + new2img[id] + '\n')