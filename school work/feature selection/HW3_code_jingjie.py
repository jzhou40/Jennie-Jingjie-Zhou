#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:05:34 2019

@author: Jingjie Zhou
"""

from scipy.io import arff
import pandas as pd
import math
from scipy.stats import zscore
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier

def pearson(x,y):
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    N = len(x)
    for i in range(N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = math.sqrt((sum_sq_x/N) - (mean_x * mean_x))
    pop_sd_y = math.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    return abs(correlation)
# pre-processing data
data = arff.loadarff('veh-prime.arff')
df = pd.DataFrame(data[0])
df['CLASS'] = df['CLASS'].map({b'car' : 1, b'noncar' : 0}).astype(int)
x = df[df.columns[0 :-1]]
y = df[df.columns[-1]]


#1 feature
dic_r = {}
for i in list(x.columns):
    dic_r[i] = pearson(x[i],y)
r_list = sorted(dic_r.items(),key = lambda kv:(kv[1],kv[0]),reverse = True)
print(r_list)

def CV_classify_acc(data_train, f=1) :
    acc = []
    neigh = KNeighborsClassifier(n_neighbors = 7)
    for i in range(int(len(data_train) / f)) :
        count = 0
        cv_test = data_train.loc[i * f :(i + 1) * f - 1, :]
        cv_train = data_train.drop(list(cv_test.index))
        x_cvtrain = cv_train.loc[:, list(cv_test.columns)[0 :-1]]
        y_cvtrain = cv_train.loc[:, 'CLASS']
        x_cvtest = cv_test.loc[:, list(cv_test.columns)[0 :-1]]
        y_cvtest = cv_test.loc[:, 'CLASS']
        neigh.fit(x_cvtrain, y_cvtrain)
        y_predict = neigh.predict(x_cvtest)
        for j in range(len(cv_test)) :
            if y_predict[j] == y_cvtest.values[j] :
                count += 1
        acc += [count / len(cv_test)]
    accuracy = mean(acc)
    return accuracy
#filter method
df.loc[:, list(df.columns)[0 :-1]] = df.loc[:, list(df.columns)[0 :-1]].apply(zscore)
col = []
filter_method = {}
for i in range(len(r_list)) :
    col.append(r_list[i][0])
    model = df[col+['CLASS']]
    accu = [CV_classify_acc(model, f = 1)]
    filter_method[tuple(col)] = accu
max_accu_feature = max(filter_method, key=filter_method.get)
max_accu = filter_method[max_accu_feature]
print(max_accu_feature)
print(max_accu)


#wrapper method
j = 0
col_w = []
accu_w = [-1,0]
total = list(df.columns)[:-1]
while accu_w[-1] > accu_w[-2] :
    pre_max = 0
    for feature in total:
        m = df[col_w+[feature]+['CLASS']]
        new = CV_classify_acc(m, f = 1)
        if new > pre_max:
            select = feature
            pre_max = new
    accu_w.append(pre_max)
    col_w.append(select)
    total.remove(select)
for q in range(len(col_w)-1):
    print(col_w[:q+1])
print(accu_w[-2])
      
        
    

    

