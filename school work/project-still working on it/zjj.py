#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:24:33 2019

@author: chengzhao
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


#child_data
df_ch = pd.read_csv('./Autism-Child-Data.csv')


df_ch0 = df_ch[df_ch["ethnicity"]=='?']
df_ch_not0 = df_ch[df_ch["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
#ethnColumns = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score",
#               "A9_Score","A10_Score",'jundice','austim','contry_of_res']
ethnColumns = ["A1_Score","A2_Score","A3_Score","A4_Score"]
df_ch_not0["ethnicity"] = df_ch_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_ch_not0[ethnColumns], df_ch_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_ch0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_ch0["ethnicity"] = ethn0Values
df_ch_zjj = df_ch0.append(df_ch_not0)
#turn class label into num
df_ch_zjj['Class/ASD'] = df_ch_zjj['Class/ASD'].replace({'NO':0,'YES':1})
#turn into one_hot code
le = LabelEncoder()
df_ch_zjj["contry_of_res"] = le.fit_transform(df_ch_zjj["contry_of_res"])
mapping_ch = dict(zip(le.classes_, range(len(le.classes_))))
#select ethnicity, country,class out
ch_zjj = df_ch_zjj[['ethnicity','contry_of_res','Class/ASD']]

##adolescent_data
df_adol = pd.read_csv('./Autism-Adolescent-Data.csv')
df_adol0 = df_adol[df_adol["ethnicity"]=='?']
df_adol_not0 = df_adol[df_adol["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["A1_Score","A2_Score","A3_Score","A4_Score"]
df_adol_not0["ethnicity"] = df_adol_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_adol_not0[ethnColumns], df_adol_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_adol0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_adol0["ethnicity"] = ethn0Values
df_adol_zjj = df_adol0.append(df_adol_not0)
#turn class label into num
df_adol_zjj['Class/ASD'] = df_adol_zjj['Class/ASD'].replace({'NO':0,'YES':1})
#turn into one_hot code
le = LabelEncoder()
df_adol_zjj["contry_of_res"] = le.fit_transform(df_adol_zjj["contry_of_res"])
mapping_adol = dict(zip(le.classes_, range(len(le.classes_))))
#select ethnicity, country,class out
adol_zjj = df_adol_zjj[['ethnicity','contry_of_res','Class/ASD']]




##adult_data
df_adult = pd.read_csv('./Autism-Adult-Data.csv')
df_adult0 = df_adult[df_adult["ethnicity"]=='?']
df_adult_not0 = df_adult[df_adult["ethnicity"]!='?']
rfModel_ethn = RandomForestRegressor()
ethnColumns = ["A1_Score","A2_Score","A3_Score","A4_Score"]
df_adult_not0["ethnicity"] = df_adult_not0.ethnicity.map({ "White-European":1, "Latino":2, "Others":3, 
          "others":3, "Black":4, 'Asian':5,'Middle Eastern ': 6, 'Pasifika':7,'South Asian':8, 'Hispanic':9,
          'Turkish':10, })
rfModel_ethn.fit(df_adult_not0[ethnColumns], df_adult_not0["ethnicity"])
ethn0Values = rfModel_ethn.predict(X= df_adult0[ethnColumns])
ethn0Values = [ round(elem) for elem in list(ethn0Values) ]
df_adult0["ethnicity"] = ethn0Values
df_adult_zjj = df_adult0.append(df_adult_not0)
#turn class label into num
df_adult_zjj['Class/ASD'] = df_adult_zjj['Class/ASD'].replace({'NO':0,'YES':1})
#turn into one_hot code
le = LabelEncoder()
df_adult_zjj["contry_of_res"] = le.fit_transform(df_adult_zjj["contry_of_res"])
mapping_adult = dict(zip(le.classes_, range(len(le.classes_))))
#select ethnicity, country,class out
adult_zjj = df_adult_zjj[['ethnicity','contry_of_res','Class/ASD']]
#
#
#
#
##visualization
##fig, ax = plt.subplots(figsize=(9.2, 10))
##plt.barh(df_adol["ethnicity"].unique(),df_adol["ethnicity"].value_counts())
##fig, ax = plt.subplots(figsize=(9.2, 10))
##plt.barh(df_adol["contry_of_res"].unique(),df_adol["contry_of_res"].value_counts())