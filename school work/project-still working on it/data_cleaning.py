# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:29:17 2019

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
######################################Weifeng age

df_Adol = pd.read_csv('Autism-Adolescent-Data.csv')
df_Adul = pd.read_csv('Autism-Adult-Data.csv')
df_Child = pd.read_csv('Autism-Child-Data.csv')

df_Adul_clean = df_Adul[df_Adul['age'] != '?']
df_Child_clean = df_Child
df_Adol_clean = df_Adol

df_Child_clean['age'].loc[df_Child_clean['age'] == '?'] = '6'

df_Child_clean[['age']] = df_Child_clean[['age']].astype('int')
df_Adol_clean[['age']] = df_Adol_clean[['age']].astype('int')
df_Adul_clean[['age']] = df_Adul_clean[['age']].astype('int')


######################################Zihan  jundice austim used_app

data_adolescent_zihan = df_Adol[['jundice','austim','used_app_before']]
data_adult_zihan = df_Adul[['jundice','austim','used_app_before']]
data_child_zihan = df_Child[['jundice','austim','used_app_before']]

# turning categorical values into numeric
data_adolescent_zihan = data_adolescent_zihan.replace(to_replace = ['yes', 'no'], value = [int(1), int(0)])
data_adult_zihan = data_adult_zihan.replace(to_replace = ['yes', 'no'], value = [int(1), int(0)])
data_child_zihan = data_child_zihan.replace(to_replace = ['yes', 'no'], value = [int(1), int(0)])

df_Child_clean[['jundice']] = data_child_zihan[['jundice']].astype('int')
df_Adul_clean[['jundice']] = data_adult_zihan[['jundice']].astype('int')
df_Adol_clean[['jundice']] = data_adolescent_zihan[['jundice']].astype('int')

df_Child_clean[['austim']] = data_child_zihan[['austim']].astype('int')
df_Adul_clean[['austim']] = data_adult_zihan[['austim']].astype('int')
df_Adol_clean[['austim']] = data_adolescent_zihan[['austim']].astype('int')

df_Child_clean[['used_app_before']] = data_child_zihan[['used_app_before']].astype('int')
df_Adul_clean[['used_app_before']] = data_adult_zihan[['used_app_before']].astype('int')
df_Adol_clean[['used_app_before']] = data_adolescent_zihan[['used_app_before']].astype('int')

##################################age_desc,gender,country,relation
#adult


#change label 'age_desc' type,,,18 and more == 1,for this label has only one attribute so get the numeric 1
df_Adul_clean['age_desc'] = df_Adul_clean['age_desc'].replace({'18 and more':1})

#change label 'gender' type,,,m == 1, f == 0
df_Adul_clean['gender'] = df_Adul_clean['gender'].replace({'m':1,'f':0})
        

#turn into one_hot code
le = LabelEncoder()
df_Adul_clean["contry_of_res"] = le.fit_transform(df_Adul_clean["contry_of_res"])
#mapping_adult = dict(zip(le.classes_, range(len(le.classes_))))

df_ck_adult0 = df_Adul_clean[df_Adul_clean["relation"]=='?']
df_ck_adult_not0 = df_Adul_clean[df_Adul_clean["relation"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_ck_adult_not0["relation"] = df_ck_adult_not0.relation.map({'Parent': 2, 'Self': 4,'self':4, 'Health care professional': 8, 'Relative': 16 , 'Others':32})

rfModel_ethn.fit(df_ck_adult_not0[ethnColumns], df_ck_adult_not0["relation"])

ethn0Values = rfModel_ethn.predict(X= df_ck_adult0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_ck_adult0["relation"] = ethn0Values
df_ck_adult_ck = df_ck_adult0.append(df_ck_adult_not0)
df_ck_adult_1 = df_ck_adult_not0['relation']


df_ck_adult_1 = df_ck_adult_not0
df_ck_adult_ck["relation"] = df_ck_adult_ck.relation

#avoid errors we get bigger range on label relation and x<3 = parent==0 , 3< x < 6 = self==1 , 6<x < 12 = Health care professional==2, 12<x < 24 = relative==3, x> 24 = others==4
df_ck_adult_ck.loc[ df_ck_adult_ck['relation'] <= 3, 'relation'] 					       = 0
df_ck_adult_ck.loc[(df_ck_adult_ck['relation'] > 3) & (df_ck_adult_ck['relation'] <= 6), 'relation'] = 1
df_ck_adult_ck.loc[(df_ck_adult_ck['relation'] > 6) & (df_ck_adult_ck['relation'] <= 12), 'relation'] = 2
df_ck_adult_ck.loc[(df_ck_adult_ck['relation'] > 12) & (df_ck_adult_ck['relation'] <= 24), 'relation'] = 3
df_ck_adult_ck.loc[ df_ck_adult_ck['relation'] > 24, 'relation'] = 4

Adult_string_kan = df_ck_adult_ck[["relation"]]
#'Parent': 0, 'Self': 1, 'Health care professional': 2, 'Relative': 3 , 'Others':4

##############child
df_Child_clean['age_desc'] = df_Child_clean['age_desc'].replace({'4-11 years':1})

#change label 'gender' type,,,m == 1, f == 0
df_Child_clean['gender'] = df_Child_clean['gender'].replace({'m':1,'f':0})

#turn into one_hot code
le = LabelEncoder()
df_Child_clean["contry_of_res"] = le.fit_transform(df_Child_clean["contry_of_res"])
#mapping_ch = dict(zip(le.classes_, range(len(le.classes_))))

df_Child_clean = df_Child_clean.iloc[:,:]
df_ck_child0 = df_Child_clean[df_Child_clean["relation"]=="?"]
df_ck_child_not0 = df_Child_clean[df_Child_clean["relation"]!="?"]
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_ck_child_not0["relation"] = df_ck_child_not0.relation.map({'Parent': 2, 'Self': 4,'self':4, 'Health care professional': 8, 'Relative': 16 })

rfModel_ethn.fit(df_ck_child_not0[ethnColumns], df_ck_child_not0["relation"])

ethn0Values = rfModel_ethn.predict(X= df_ck_child0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_ck_child0["relation"] = ethn0Values
df_ck_child_ck = df_ck_child0.append(df_ck_child_not0)
df_ck_child_1 = df_ck_child_not0['relation']



df_ck_child_1 = df_ck_child_not0
df_ck_child_ck["relation"] = df_ck_child_ck.relation

#avoid errors we get bigger range on label relation and x<3 = parent==0 , 3< x < 6 = self==1 , 6<x < 12 = Health care professional==2, 12<x = relative==3
df_ck_child_ck.loc[ df_ck_child_ck['relation'] <= 3, 'relation'] 					       = 0
df_ck_child_ck.loc[(df_ck_child_ck['relation'] > 3) & (df_ck_child_ck['relation'] <= 6), 'relation'] = 1
df_ck_child_ck.loc[(df_ck_child_ck['relation'] > 6) & (df_ck_child_ck['relation'] <= 12), 'relation'] = 2
df_ck_child_ck.loc[(df_ck_child_ck['relation'] > 12) , 'relation'] = 3

Child_string_kan = df_ck_child_ck[['relation']]

#'Parent': 0, 'Self': 1, 'Health care professional': 2, 'Relative': 3

####################adol
        
df_Adol_clean['age_desc'] = df_Adol_clean['age_desc'].replace({'12-16 years':1,'12-15 years':0})

#change label 'gender' type,,,m == 1, f == 0
df_Adol_clean['gender'] = df_Adol_clean['gender'].replace({'m':1,'f':0})
#turn into one_hot code
le = LabelEncoder()
df_Adol_clean["contry_of_res"] = le.fit_transform(df_Adol_clean["contry_of_res"])
#mapping_adol = dict(zip(le.classes_, range(len(le.classes_))))

df_ck_adolescent0 = df_Adol_clean[df_Adol_clean["relation"]=='?']
df_ck_adolescent_not0 = df_Adol_clean[df_Adol_clean["relation"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_ck_adolescent_not0["relation"] = df_ck_adolescent_not0.relation.map({'Parent': 2, 'Self': 4,'self':4, 'Health care professional': 8, 'Relative': 16 , 'Others':32})

rfModel_ethn.fit(df_ck_adolescent_not0[ethnColumns], df_ck_adolescent_not0["relation"])

ethn0Values = rfModel_ethn.predict(X= df_ck_adolescent0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_ck_adolescent0["relation"] = ethn0Values
df_ck_adolescent_ck = df_ck_adolescent0.append(df_ck_adolescent_not0)
df_ck_adolescent_1 = df_ck_adolescent_not0['relation']


df_ck_adolescent_1 = df_ck_adolescent_not0
df_ck_adolescent_ck["relation"] = df_ck_adolescent_ck.relation

#avoid errors we get bigger range on label relation and x<3 = parent==0 , 3< x < 6 = self==1 , 6<x < 12 = Health care professional==2, 12<x < 24 = relative==3, x> 24 = others==4
df_ck_adolescent_ck.loc[ df_ck_adolescent_ck['relation'] <= 3, 'relation'] 					       = 0
df_ck_adolescent_ck.loc[(df_ck_adolescent_ck['relation'] > 3) & (df_ck_adolescent_ck['relation'] <= 6), 'relation'] = 1
df_ck_adolescent_ck.loc[(df_ck_adolescent_ck['relation'] > 6) & (df_ck_adolescent_ck['relation'] <= 12), 'relation'] = 2
df_ck_adolescent_ck.loc[(df_ck_adolescent_ck['relation'] > 12) & (df_ck_adolescent_ck['relation'] <= 24), 'relation'] = 3
df_ck_adolescent_ck.loc[ df_ck_adolescent_ck['relation'] > 24, 'relation'] = 4

Adolescent_string_kan = df_ck_adolescent_ck[['relation']]
#'Parent': 0, 'Self': 1, 'Health care professional': 2, 'Relative': 3 , 'Others':4


df_Adol_clean = df_Adol_clean.drop(['relation'], axis=1)
df_Child_clean = df_Child_clean.drop(['relation'], axis=1)
df_Adul_clean = df_Adul_clean.drop(['relation'], axis=1)

df_Adol_clean = pd.merge(df_Adol_clean, Adolescent_string_kan, left_index=True, right_index=True)
df_Child_clean = pd.merge(df_Child_clean, Child_string_kan, left_index=True, right_index=True)
df_Adul_clean = pd.merge(df_Adul_clean, Adult_string_kan, left_index=True, right_index=True)


#########################################zjj

#child_data

#ethnicity
df_ch0 = df_Child_clean[df_Child_clean["ethnicity"]=='?']
df_ch_not0 = df_Child_clean[df_Child_clean["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_ch_not0["ethnicity"] = df_ch_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_ch_not0[ethnColumns], df_ch_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_ch0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_ch0["ethnicity"] = ethn0Values
df_ch_zjj = df_ch0.append(df_ch_not0)
#select ethnicity, country,class out
ch_zjj = df_ch_zjj[['ethnicity']]


##adolescent_data

df_adol0 = df_Adol_clean[df_Adol_clean["ethnicity"]=='?']
df_adol_not0 = df_Adol_clean[df_Adol_clean["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_adol_not0["ethnicity"] = df_adol_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_adol_not0[ethnColumns], df_adol_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_adol0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_adol0["ethnicity"] = ethn0Values
df_adol_zjj = df_adol0.append(df_adol_not0)
#select ethnicity, country,class out
adol_zjj = df_adol_zjj[['ethnicity']]


##adult_data


#turn class label into num

df_adult0 = df_Adul_clean[df_Adul_clean["ethnicity"]=='?']
df_adult_not0 = df_Adul_clean[df_Adul_clean["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["age",'gender','jundice','austim','contry_of_res','used_app_before','age_desc']
df_adult_not0["ethnicity"] = df_adult_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_adult_not0[ethnColumns], df_adult_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_adult0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_adult0["ethnicity"] = ethn0Values
df_adult_zjj = df_adult0.append(df_adult_not0)

#select ethnicity, country,class out
adult_zjj = df_adult_zjj[['ethnicity']]

df_Adol_clean = df_Adol_clean.drop(['ethnicity'], axis=1)
df_Child_clean = df_Child_clean.drop(['ethnicity'], axis=1)
df_Adul_clean = df_Adul_clean.drop(['ethnicity'], axis=1)

df_Adol_clean = pd.merge(df_Adol_clean, adol_zjj, left_index=True, right_index=True)
df_Child_clean = pd.merge(df_Child_clean, ch_zjj, left_index=True, right_index=True)
df_Adul_clean = pd.merge(df_Adul_clean, adult_zjj, left_index=True, right_index=True)

df_Adul_clean['Class/ASD'] = df_Adul_clean['Class/ASD'].replace({'NO':0,'YES':1})
df_Adol_clean['Class/ASD'] = df_Adol_clean['Class/ASD'].replace({'NO':0,'YES':1})
df_Child_clean['Class/ASD'] = df_Child_clean['Class/ASD'].replace({'NO':0,'YES':1})



#delete outlier instance
df_Adul_new1 = df_Adul_clean.iloc[0:53,:]
df_Adul_new2 = df_Adul_clean.iloc[53:,:]
frame_adult = [df_Adul_new1,df_Adul_new2]
df_Adul_clean = pd.concat(frame_adult)
#
#df_Adol_clean.to_csv (r'Autism-Adolescent-Data-Clean.csv',index=False)
#df_Adul_clean.to_csv (r'Autism-Adult-Data-Clean.csv',index=False)
#df_Child_clean.to_csv (r'Autism-Child-Data_Clean.csv',index=False)
drop_list1 = ["age_desc",'id']
df_Adul_clean = df_Adul_clean.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

y1 = df_Adul_clean['Class/ASD']
x1 = df_Adul_clean.drop(['Class/ASD'],axis =1)
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3,random_state = 20)

#Decision Tree
cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(x_train))
depth_range = range(1, max_attributes + 1)
train = pd.concat([x_train,y_train],axis = 1)
train = train.reset_index(drop=True)
# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices
        model = tree_model.fit(X = f_train.drop(['Class/ASD'], axis=1), 
                               y = f_train["Class/ASD"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Class/ASD'], axis=1), 
                                y = f_valid["Class/ASD"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
#decision_tree = tree.DecisionTreeClassifier()
#dft = decision_tree.fit(x_train, y_train)
#Y_pred = decision_tree.predict(x_test)
#acc_decision_tree = round(decision_tree.score(x_test, y_test) * 100, 2)
#tree.plot_tree(dft.fit(x_train, y_train)) 
#random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
acc_random_forest = round(random_forest.score(x_test, y_test) * 100, 2)
#Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
acc_perceptron = round(perceptron.score(x_test, y_test) * 100, 2)











