#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:32:36 2019

@author: chengzhao
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re 
wordnet_lemmatizer = WordNetLemmatizer()

emoticons_str = r""" 
    (?:
    [:=;] # 眼睛 
    [oO\-]? # ⿐⼦ 
    [D\)\]\(\]/\\OpP] # 嘴 
    )""" 
regex_str = [
        emoticons_str, 
        r'<[^>]+>', # HTML tags 
        r'(?:@[\w_]+)', # @某⼈ 
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # 话题标签 
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs 
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # 数字 
        r"(?:[a-z][a-z'\-_]+[a-z])", # 含有 - 和 ‘ 的单词 
        r'(?:[\w_]+)', # 其他 
        r'(?:\S)' # 其他
]
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE) 
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
    return tokens_re.findall(s)
def get_wordnet_pos(word):         #找对应的pos_tag
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def preprocess(s,lowercase = False):
    tokens = tokenize(s) 
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    token_list = []
    for word in tokens:
        if emoticon_re.search(word):                            #有表情符号先提取
            token_list.append(word)
        else:
            if lowercase:
                word.lower()
            token_list.append(wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word)))   #每个词做对应处理
    return {word: True for word in token_list} # return:# {'this': True, 'is':True, 'a':True, 'good':True, 'book':True}
#    return {word: True for word in s.lower().split()}

pos_data = [] 
with open('./rt-polarity.pos', encoding='latin-1') as f:
    for line in f:
        pos_data.append([preprocess(line,True), 'pos'])
neg_data = []
with open('./rt-polarity.neg', encoding='latin-1') as f:
    for line in f:
        neg_data.append([preprocess(line,True), 'neg'])
training_data = pos_data[:5000]+neg_data[:5000] 
testing_data = pos_data[5000:]+neg_data[5000:]
model = NaiveBayesClassifier.train(training_data)
 
correct = 0
for data in testing_data:
    content = data[0]
    if model.classify(content) == data[1]:
        correct +=1
pro = correct/len(testing_data)    #probability of correctness
print(pro) 
