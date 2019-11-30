# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:46:48 2019

@author: melan
"""
import numpy as np
import pandas as pd
from pandas import DataFrame 
#Dealing with null value and date feature in training set
filepath="train.csv"
train=pd.read_csv(filepath,header=0,encoding='gbk')
train=train.dropna()
train=train.reset_index(drop=True)
print(train)
for i in range(0,train.shape[0]):
    train.loc[i,'release_date']=train.loc[i,'release_date'].replace(", ","-").replace(" ","-")
train.to_csv("train.csv",index=False)
#Dealing with null value in testing set
filepath="test.csv"
test=pd.read_csv(filepath,header=0,encoding='gbk')
test.fillna(round(test.median()),inplace=True)
test["purchase_date"].fillna(axis=0,method='ffill',inplace=True)
test.to_csv("test.csv",index=False)