# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:31:08 2019

@author: melan
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
filepath=r"train_target2.csv"
data=pd.read_csv(filepath,header=0,encoding='gbk')
x=data[['is_free','price','genres','categories','tags','total_positive_reviews','total_negative_reviews','postive_rate','month_p','year_p','year_r','gap_between_date']]
y=data[['playtime_forever']] 
# =============================================================================
# x=data[['is_free','price','genres','categories','tags','total_positive_reviews','total_negative_reviews','postive_rate','month_p','year_p','year_r','gap_between_date']]
# y=data[['playtime_forever']] 
# =============================================================================
DT_scores=0
lgb_scores=0
RF_scores=0
xgb_scores=0
ada_scores=0  
    
kf = KFold(10, shuffle=True, random_state=42).get_n_splits(x)     
          
model_DT=DecisionTreeRegressor()
DT_scores = cross_val_score(model_DT,x.values, y.values, cv=kf,scoring='neg_mean_squared_error').mean()
model_DT.fit(x.values, y.values)

model_lgb = lgb.LGBMRegressor()
lgb_scores = cross_val_score(model_lgb,x.values, y.values, cv=kf,scoring='neg_mean_squared_error').mean()
model_lgb.fit(x.values, y.values)
 
model_RF=RandomForestRegressor()
RF_scores = cross_val_score(model_RF,x.values, y.values, cv=kf,scoring='neg_mean_squared_error').mean()
model_RF.fit(x.values, y.values)

model_ada=AdaBoostRegressor()
ada_scores = cross_val_score(model_ada,x.values, y.values, cv=kf,scoring='neg_mean_squared_error').mean()
model_ada.fit(x.values, y.values)
    
# =============================================================================
#     model_LR = LinearRegression()
#     LR_scores + = cross_val_score(model_LR,x.values, y.values, cv=10,scoring='neg_mean_squared_error').mean()  
#     model_LR.fit(x.values, y.values)
# =============================================================================
    
model_xgb = xgb.XGBRegressor()
xgb_scores = cross_val_score(model_xgb,x.values, y.values, cv=kf,scoring='neg_mean_squared_error').mean()
model_xgb.fit(x.values, y.values)

# =============================================================================
# joblib.dump(model_RF,'RF.pkl') 
# joblib.dump(model_DT,'DT.pkl') 
# joblib.dump(model_DT,'LR.pkl') 
# joblib.dump(model_lgb,'lgb.pkl') 
# joblib.dump(model_xgb,'xgb.pkl') 
# =============================================================================

print('rmse of RF:', np.sqrt(-float(RF_scores)), 'degrees.')
print(model_RF.feature_importances_)
print('rmse of DT:',np.sqrt(-float(DT_scores)), 'degrees.')
print('rmse of xgb:',np.sqrt(-float(xgb_scores)), 'degrees.')
print('rmse of lgb:',np.sqrt(-float(lgb_scores)), 'degrees.')
print('rmse of ada:',np.sqrt(-float(ada_scores)), 'degrees.')


# =============================================================================
# parameters = {'n_estimators': range(10,200,20), 'max_features' : range(2,10,1),    'max_depth' : range(2,10,1)} 
# rf = RandomForestRegressor()
# grid_obj = GridSearchCV(rf, parameters, cv=5,scoring='neg_mean_squared_error')
# grid_fit = grid_obj.fit(x.values, y.values)
# best_RF = grid_fit.best_estimator_  
# #best_clf.fit(train,labels)
# print(grid_fit.best_score_)
# joblib.dump(best_RF,'RF(best).pkl')
# 
# parameters = {'max_depth':list(range(3,10,2)),
#  'min_child_weight':list(range(1,6,2)),
#  'n_estimators': list(range(10,200,20)),
#  'learning_rate': [0.1,0.01]} 
# xgb = xgb.XGBRegressor()
# grid_obj = GridSearchCV(xgb, parameters, cv=5,scoring='neg_mean_squared_error')
# grid_fit = grid_obj.fit(x.values, y.values)
# best_xgb = grid_fit.best_estimator_  
# print(grid_fit.best_estimator_ )
# print(grid_fit.best_score_)
# joblib.dump(best_xgb,'XGBoost(best).pkl')
# 
# =============================================================================
