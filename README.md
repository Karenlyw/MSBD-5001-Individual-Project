# MSBD-5001-Kaggle
5001 individual project  (https://www.kaggle.com/c/msbd5001-fall2019)
Language : python3.7  
Package:numpy pandas datetime xgboost lightgbm sklearn  
For each submission, run as follow: Feature engineering.py   Model.py  Prediction.py  
## 1.Basic preprocess(Basic Preprocess.py)
   First, I dealt with null value and date forms in both training and testing set. Then, I unified the date form by setting all of them to "year/month/day" form in Excel.(test.csv and train.csv are the results of this code)
## 2.Feature engineering(Feature engineering.py)
   For date features, I extracted year, month and day respectively and only kept purchase_month, purchase_year, release_year as well as adding the gap between purchase and release date.
   For categorical features(genres, categories and tags), I chose target encoding. Since the dataset is quite small, onehot encoding may lead to overfitting due to relatively high dimension of features. 
   I'm not sure whether target encoding will also cause overfitting, so I tried smooth(submission1) and non_smooth(submission2) target encoding.
   I also added "postive_review_rate"feature. 
   (test_target/test_target2.csv and train_target/train_target2.csv are the results of this code)
## 3.Training model(Model.py)
   After comparing several models using cross validation, I chose xgboost and random forest as my baseline models and did fine tuning on them. By using GridSearchcv, I got two models.(XGBoost(best).pkl and RF(best).pkl )
## 4.Prediction(Prediction.py)
   3 prediction results are generated from XGBoost(prediction(xgb).csv), Random forest(prediction(rf).csv) and the combination(mean) of this two(prediction(ensemble).csv). Noticing there is an outlier in the public leaderboard, I changed the value of it manually. Of course it will not affect the private result. 
   My final submissions are prediction(xgb).csv in file Submission1 and prediction(ensemble).csv in file Submission2.
