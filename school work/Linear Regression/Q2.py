#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:03:10 2019

@author: chengzhao
"""

import pandas as pd
import numpy as np
def data(address):
    data_t = pd.read_csv(address)
    n = len(data_t)/10
    data_t.insert(0,'x0',1)
    return (data_t,n)
def mse(lamda,x_train,y_train,x_test,y_test):
        w = np.linalg.inv(np.dot(x_train.T,x_train)+lamda*np.eye(x_train.shape[1])).dot(x_train.T).dot(y_train)
        m_test = 1/len(x_test)*sum((x_test.dot(w)-y_test)**2)
        return m_test
def Q2(data_train,data_test,n):
    ave_mse=[]
    for lamda in range(151):
        sum_mse = 0
        for i in range(10):
            cv_test = data_train.loc[i*n:(i+1)*n-1,:]
            cv_train = data_train
            cv_train=cv_train.append(cv_test)
            cv_train=cv_train.append(cv_test)
            cv_train = cv_train.drop_duplicates(list(cv_train.columns),keep=False)
            x_cvtrain = cv_train.loc[:,list(cv_test.columns)[0:-1]]
            y_cvtrain = cv_train.loc[:,'y']
            x_cvtest = cv_test.loc[:,list(cv_test.columns)[0:-1]]
            y_cvtest = cv_test.loc[:,'y']
            sum_mse +=mse(lamda,x_cvtrain,y_cvtrain,x_cvtest,y_cvtest)
        ave_mse.append(sum_mse/10)
    min_lamda = ave_mse.index(min(ave_mse))
    test_mse=mse(min_lamda,data_train.loc[:,list(data_train.columns)[0:-1]],data_train.loc[:,'y'],data_test.loc[:,list(data_test.columns)[0:-1]],data_test.loc[:,'y'])
    return (min_lamda,test_mse)
print(Q2(data('./train-100-10.csv')[0],data('./test-100-10.csv')[0],data('./train-100-10.csv')[1]))
print(Q2(data('./train-100-100.csv')[0],data('./test-100-100.csv')[0],data('./train-100-100.csv')[1]))
print(Q2(data('./train-1000-100.csv')[0],data('./test-1000-100.csv')[0],data('./train-1000-100.csv')[1]))
print(Q2(data('./train-50(1000)- 100.csv')[0],data('./test-1000-100.csv')[0],data('./train-50(1000)- 100.csv')[1]))
print(Q2(data('./train-100(1000)-100.csv')[0],data('./test-1000-100.csv')[0],data('./train-100(1000)-100.csv')[1]))
print(Q2(data('./train-150(1000)-100.csv')[0],data('./test-1000-100.csv')[0],data('./train-150(1000)-100.csv')[1]))