#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:38:38 2019

@author: chengzhao
"""
import pandas as pd
import numpy as np
from scipy import stats
train_data = pd.read_csv('./spam_train.csv')
test_data = pd.read_csv('./spam_test.csv')


def distance(a,b):
    return np.linalg.norm(a - b)

def knn(k,train_data,test_data,znorm):
    train_x = train_data.iloc[:,:-1].values
    train_y = train_data.iloc[:,-1].values
    test_x = test_data.iloc[:,1:-1].values
    test_y = test_data.iloc[:,-1].values
    if znorm=='Y':
        train_x=stats.zscore(train_x)
        test_x=stats.zscore(test_x)
    accu = 0
    for i in range(len(test_data)):
        dist =[]
        for j in range(len(train_data)):
            dist+=[(distance(train_x[j],test_x[i]),j,train_y[j])]
        dist.sort(key=lambda x:x[0])
        neighbor = dist[:k]
        vote1,vote0=0,0
        for n in neighbor:
            if n[2]==1:
                vote1+=1
            elif n[2]==0:
                vote0+=1
        if vote1>vote0:
            plabel = 1
        else:
            plabel = 0
        if plabel==test_y[i]:
            accu +=1
    return accu/len(test_data)
K=[1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
res_n =[]
for k in K:
    res_n.append(knn(k,train_data,test_data,'N'))   
print(res_n)
res_zn =[]
for k in K:
    res_zn.append(knn(k,train_data,test_data,'Y'))
print(res_zn)
train_xz = stats.zscore(train_data.iloc[:,:-1].values)
train_y = train_data.iloc[:,-1].values
test_xz = stats.zscore(test_data.iloc[:,1:-1].values)

def knn_label(K,X,train_xz,train_y):   #X single test data,K array
    dist =[]
    label=[]
    for j in range(len(train_data)):
        dist+=[(distance(train_xz[j],X),j,train_y[j])]
    dist.sort(key=lambda x:x[0])
    for k in K:
        neighbor = dist[:k]
        vote1,vote0=0,0
        for n in neighbor:
            if n[2]==1:
                vote1+=1
            elif n[2]==0:
                vote0+=1
        if vote1>vote0:
            plabel='spam'
        else:
            plabel='no'  
        label.append(plabel)
    return label
for i in range(50):
    output = [test_data.iloc[i,0]]+knn_label(K,test_xz[i],train_xz,train_y)
    print(output)
 
# def knn(k, test_x, train_x, train_y) :  # X single test data,K array
#    dist = []
#    label = []
#    train_x.apply(zscore)
#    test_x.apply(zscore)
#    for i in range(len(test_x)) :
#        for j in range(len(train_x)) :
#            dist += [(np.linalg.norm(train_x.iloc[j] - test_x.iloc[i]), j, train_y.iloc[j][0])]
#        
#        dist.sort(key = lambda x : x[0])
#        neighbor = dist[:k]
#        print(neighbor)
#        vote1, vote0 = 0, 0
#        for n in neighbor :
#            if n[2] == 1 :
#                vote1 += 1
#                print('vote1:',vote1)
#            elif n[2] == 0 :
#                vote0 += 1
#                print('vote0:',vote0)
#        if vote1 > vote0 :
#            plabel = 1
#        else :
#            plabel = 0
#            
#        label.append(plabel)
#    return label
# p = knn(7, x2, x, y)            
            
            
            
            
            
            
    