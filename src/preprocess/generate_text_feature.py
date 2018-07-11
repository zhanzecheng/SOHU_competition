# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/9 下午4:25
# @Author  : zhanzecheng
# @File    : generate_text_feature.py
# @Software: PyCharm
"""
import pickle
import numpy as np
import os
import sys
import pandas as pd
import re

# check有多少个微信的字样
def check_wechat(string):
    string = string.replace(' ', '')
    rule = '微信'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

def check_qq(string):
    string = string.replace(' ', '')
    rule = 'qq'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少折的字样
def check_sale(string):
    string = string.replace(' ', '')
    rule = "(\\d|[一、二、三、四、五、六、七、八、九、十])折"
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length


# check有多少个广告的字样
def check_advertise(string):
    string = string.replace(' ', '')
    rule = '广告'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少个送礼包的字样
def check_give(string):
    string = string.replace(' ', '')
    rule = u"送*(礼品|礼包)"
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少个关注的字样
def check_follow(string):
    string = string.replace(' ', '')
    rule = '关注|加入'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少阅读原文 原文阅读 点击此处
def check_readhere(string):
    string = string.replace(' ', '')
    rule = '阅读原文|原文阅读|点击此处|请点这里|点击这里|快戳|请戳|订阅'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有带ID的字眼
def check_id(string):
    string = string.replace(' ', '')
    rule = 'id'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少个原创的字样
def check_origin(string):
    string = string.replace(' ', '')
    rule = '原创'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check有多少个来源的字样
def check_from(string):
    string = string.replace(' ', '')
    rule = '来源'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check 仅 只要 只需
def check_only_need(string):
    string = string.replace(' ', '')
    rule = '仅需|仅要|只要|只需'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check !
def check_sock(string):
    string = string.replace(' ', '')
    rule = '!|！'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check ?
def check_why(string):
    string = string.replace(' ', '')
    rule = '\?|？'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# .com
def check_com(string):
    string = string.replace(' ', '')
    rule = '.com|.cn|.org'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# 手机号和电话号码
def check_phone(string):
    string = string.replace(' ', '')
    rule = u"((?<!\\d)1[3458]\\d{9})|([\\d{3}.]?\\d{8})"
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

# check 免费
def check_free(string):
    string = string.replace(' ', '')
    rule = '免费'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

def check_money(string):
    string = string.replace(' ', '')
    rule = u"([\\d]+(元|钱))|(($|¥)[\\d]+)"
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

def get_file(origin, target):
    os.system('cp ' + origin + ' ' + target)



# check 公众号
def check_public(string):
    string = string.replace(' ', '')
    rule = '公众号|公众平台'
    pattern = re.compile(rule)
    match = pattern.finditer(string)
    length = 0
    if match != None:
        for i in match:
            length += 1
    return length

if __name__ == '__main__':


    ORIGIN_FILE = os.path.join('../../data/', sys.argv[1])
    SAVEFILE = os.path.join('../../data/', sys.argv[1].replace('.txt', '.csv'))

    print('------------> generate text features')



    # 得到id对应的text
    ID_TEXT = {}
    with open(ORIGIN_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            id = line[0]
            ID_TEXT[id] = line[1:]

    # 这里我们把新的特征给加进去
    id = []
    text = []
    for i, t in ID_TEXT.items():
        id.append(i)
        text.append(t)

    wechat_list = []
    for i in text:
        i = list(map(str.lower, i))
        wechat = sum(list(map(check_wechat, i)))
        wechat_list.append(wechat)

    qq_list = []
    for i in text:
        i = list(map(str.lower, i))
        qq = sum(list(map(check_qq, i)))
        qq_list.append(qq)

    sale_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_sale, i)))
        sale_list.append(sale)

    advertise_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_advertise, i)))
        advertise_list.append(sale)

    give_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_give, i)))
        give_list.append(sale)

    follow_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_follow, i)))
        follow_list.append(sale)

    readhere_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_readhere, i)))
        readhere_list.append(sale)

    id_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_id, i)))
        id_list.append(sale)

    origin_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_origin, i)))
        origin_list.append(sale)

    from_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_from, i)))
        from_list.append(sale)

    only_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_only_need, i)))
        only_list.append(sale)

    com_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_com, i)))
        com_list.append(sale)

    phone_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_phone, i)))
        phone_list.append(sale)

    free_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_free, i)))
        free_list.append(sale)

    money_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_money, i)))
        money_list.append(sale)

    public_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_public, i)))
        public_list.append(sale)

    why_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_why, i)))
        why_list.append(sale)

    sock_list = []
    for i in text:
        i = list(map(str.lower, i))
        sale = sum(list(map(check_sock, i)))
        sock_list.append(sale)

    df = pd.DataFrame({'id':id, 'wechat_list':wechat_list, 'qq_list':qq_list, 'sock_list': sock_list,'why_list' : why_list,   'sale_list':sale_list, 'advertise_list':advertise_list, 'give_list':give_list, 'follow_list': follow_list, 'readhere_list': readhere_list, 'id_list': id_list, 'origin_list':origin_list, 'from_list':from_list, 'only_list':only_list, 'com_list':com_list, 'phone_list':phone_list, 'free_list':free_list, 'money_list':money_list, 'public_list':public_list})
    df.to_csv(SAVEFILE, index=False)

    print('------------> done')
