# -*- coding: utf-8 -*-
"""
# @Time    : 2018/7/5 下午10:23
# @Author  : zhanzecheng
# @File    : parse_html.py
# @Software: PyCharm
"""
import codecs
import sys
import re
import os
import HTMLParser
# TODO:这里只是把title单纯的加入进去
# 可能p标签也需要修改 done

'''
这个脚本是要用python2来执行的
把文本信息从html里面提取出来
'''
html_parser = HTMLParser.HTMLParser()
space_pat = re.compile(r'\\t|\\n', re.S)
p_pat = re.compile(r'(<p(>| ))|<br>|<br/>', re.S)
sc_tag_pat = re.compile(r'<[^>]+>', re.S)
multi_space_pat = re.compile(r' +', re.S)
space_between = re.compile(r'> +<', re.S)


def str_q2b(s):

    res = ""
    for u in s:

        c = ord(u)
        if c == 12288:
            c = 32
        elif 65281 <= c <= 65374:
            c -= 65248

        res += unichr(c)

    return res



def html_filter(content):

    s1 = space_pat.sub('', content).replace(r'\r', '')
    s1 = space_between.sub('><', s1)
    s2 = p_pat.sub(lambda x: '\n' + x.group(0), s1)
    # s2 = s2.replace('\n', '')
    s3 = sc_tag_pat.sub('', s2).strip()
    s4 = html_parser.unescape(s3.decode('utf8')).encode('utf8')
    s5 = str_q2b(s4.decode('utf8')).encode('utf8').replace('\xc2\xa0', ' ')
    content = multi_space_pat.sub(' ', s5)
    content = content.split('\n')
    string = ""
    firstFlag = True
    for c in content:
        if c != ' ' and c != '':
            if firstFlag:
                string += c.strip()
                firstFlag = False
            else:
                string = string + '\t' + c.strip()
    content = string.split('\t')
    return content



if __name__ == '__main__':
    '''
    usage python2 parse_html.py XXXX.txt
    '''
    print('------------> begin to parse train html')
    SAVENAME = os.path.join('../../data/', 'News_to_train_text.txt')
    result = {}
    with codecs.open('../../data/News_info_train.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            result[line[0]] = html_filter(line[1])

    with codecs.open(SAVENAME, 'w') as f:
        for key in result.keys():
            lines = result[key]
            line = ""
            if len(lines) < 1:
                print(key)
            for count, da in enumerate(lines):
                if count == 0:
                    line += da.strip()
                else:
                    line = line + '\t' + da.strip()
            f.write(key + '\t' + line + '\n')
    print('------------> done')

    print('------------> begin to parse test html')
    SAVENAME = os.path.join('../../data/', 'News_to_test_text.txt')
    result = {}
    with codecs.open('../../data/News_info_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            result[line[0]] = html_filter(line[1])

    with codecs.open(SAVENAME, 'w') as f:
        for key in result.keys():
            lines = result[key]
            line = ""
            if len(lines) < 1:
                print(key)
            for count, da in enumerate(lines):
                if count == 0:
                    line += da.strip()
                else:
                    line = line + '\t' + da.strip()
            f.write(key + '\t' + line + '\n')
    print('------------> done')