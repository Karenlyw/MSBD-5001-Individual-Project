# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:29:45 2019

@author: melan
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from functools import reduce
filepath=r"test_target.csv"
data=pd.read_csv(filepath,header=0,encoding='gbk')
x=data[['is_free','price','genres','categories','tags','total_positive_reviews','total_negative_reviews','postive_rate','month_p','year_p','year_r','gap_between_date']]
model_xgb = joblib.load('XGBoost(best).pkl')
model_rf=joblib.load('RF(best).pkl')
pred_y_xgb=model_xgb.predict(x.values)
pred_y_rf=model_rf.predict(x.values)
for i in range(0,len(pred_y_xgb)):
    if pred_y_xgb[i]<0:
        pred_y_xgb[i]=0
print(pred_y_xgb)
print(pred_y_rf)
pred_y_ensemble=(pred_y_xgb+pred_y_rf)/2
print(pred_y_ensemble)
df=pd.DataFrame({"id":data.index,"playtime_forever":pred_y_xgb})
df.to_csv('prediction(xgb).csv',encoding='gbk',index=False)  
df2=pd.DataFrame({"id":data.index,"playtime_forever":pred_y_ensemble})
df2.to_csv('prediction(ensemble).csv',encoding='gbk',index=False)  
df3=pd.DataFrame({"id":data.index,"playtime_forever":pred_y_rf})
df3.to_csv('prediction(rf).csv',encoding='gbk',index=False)  