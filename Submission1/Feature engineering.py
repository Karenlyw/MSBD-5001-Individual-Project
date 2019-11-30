# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:28:25 2019

@author: melan
"""
import numpy as np
import pandas as pd
from pandas import DataFrame 
import datetime as dt

def to_lower(df):
    df['genres']=df['genres'].str.lower()
    df['categories']=df['categories'].str.lower()
    df['tags']=df['tags'].str.lower()
    return df
def target_computation(df):
# Extract the genres\categories\tags list from both training and testing sets and find the intersection of them.
    genres=""
    categories=""
    tags=""
    for i in range(0,df.shape[0]):
        genres=genres+df.loc[i,'genres']+','
        categories=categories+df.loc[i,'categories']+','
        tags=tags+df.loc[i,'tags']+','
    genres=genres.rstrip(',').split(',')
    genres_avg={}
    categories=categories.rstrip(',').split(',')
    categories_avg={}
    tags=tags.rstrip(',').split(',')
    tags_avg={}
    
    for genre in genres:
        m=0
        score=0
        for i in range(0,df.shape[0]):
            if genre in df.loc[i,'genres']:
                score+=df.loc[i,'playtime_forever']
                m=m+1
        genres_avg[genre]=float((score+50*np.mean(df['playtime_forever']))/(m+50))  #Smooth the avg by including the avg time of all games
    for category in categories:
        m=0
        score=0
        for i in range(0,df.shape[0]):
            if category in df.loc[i,'categories']:
                score+=df.loc[i,'playtime_forever']
                m=m+1
        categories_avg[category]=float((score+50*np.mean(df['playtime_forever']))/(m+50)) 
    for tag in tags:
        m=0
        score=0
        for i in range(0,df.shape[0]):
            if tag in df.loc[i,'tags']:
                score+=df.loc[i,'playtime_forever']
                m=m+1
        tags_avg[tag]=float((score+50*np.mean(df['playtime_forever']))/(m+50)) 
    return genres_avg,categories_avg,tags_avg
def feature_preprocess(df,genres_avg,categories_avg,tags_avg):
    for i in range(0,df.shape[0]):
        # Extract "postive rate" feature
        if df.loc[i,'total_positive_reviews']+df.loc[i,'total_negative_reviews']==0:
            df.loc[i,'postive_rate']=0
        else:
            df.loc[i,'postive_rate']=df.loc[i,'total_positive_reviews']/(df.loc[i,'total_positive_reviews']+df.loc[i,'total_negative_reviews'])
        if df.loc[i,'is_free']==True:
            df.loc[i,'is_free']=1
        else:
            df.loc[i,'is_free']=0
        # Target encoding(genres+categories+tags)
        a=0
        b=0
        c=0
        for j in genres_avg.keys():
            if j in df.loc[i,'genres']:
                a=a+genres_avg[j]   
        df.loc[i,'genres']=float(a/len( df.loc[i,'genres'].split(',')))
        for p in categories_avg.keys():
            if p in df.loc[i,'categories']:
                b=b+categories_avg[p]
        df.loc[i,'categories']=float(b/len( df.loc[i,'categories'].split(',')))      
        for q in tags_avg.keys():
            if q in df.loc[i,'tags']:
                c=c+tags_avg[q]
        df.loc[i,'tags']=float(c/len( df.loc[i,'tags'].split(',')))
        # Extract time features
        df.loc[i,'month_p']=df.loc[i,'purchase_date'].split('/')[1]
        df.loc[i,'year_p']=df.loc[i,'purchase_date'].split('/')[0]
        df.loc[i,'year_r']=df.loc[i,'release_date'].split('/')[0]
# =============================================================================
#         if int(df.loc[i,'year_p'])>int(df.loc[i,'year_r']):
#             df.loc[i,'gap_between_year']=int(df.loc[i,'year_p'])-int(df.loc[i,'year_r']) 
#         else:
#             df.loc[i,'gap_between_year']=0
# =============================================================================
        date1=dt.datetime.strptime(df.loc[i,'purchase_date'],"%Y/%m/%d") 
        date2=dt.datetime.strptime(df.loc[i,'release_date'],"%Y/%m/%d") 
        df.loc[i,'gap_between_date']=(date1-date2).days
 #       df.loc[i,'day_p']=df.loc[i,'purchase_date'].split('/')[2]
       # df.loc[i,'month_r']=df.loc[i,'release_date'].split('/')[1]
        #df.loc[i,'day_r']=df.loc[i,'release_date'].split('/')[2]

    del df['purchase_date']
    del df['release_date']
#    del df['single-player']# Without distinction
    return df  

if __name__ == "__main__":
    filepath="train.csv"
    ori_train=pd.read_csv(filepath,header=0,encoding='gbk')
    filepath2="test.csv"
    ori_test=pd.read_csv(filepath2,header=0,encoding='gbk')
    train_v1=to_lower(ori_train)
    test_v1=to_lower(ori_test)
    genres_avg,categories_avg,tags_avg=target_computation(train_v1)
    new_train=feature_preprocess(train_v1,genres_avg,categories_avg,tags_avg)
    new_test=feature_preprocess(test_v1,genres_avg,categories_avg,tags_avg)
    new_train.to_csv('train_target.csv',encoding='gbk',index=False)
    new_test.to_csv('test_target.csv',encoding='gbk',index=False)
