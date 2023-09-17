# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:45:05 2019

@author: Administrator
"""
import pandas as pd
import scipy
te= pd.concat((cossim1,word2vecsim1,jaccardsim1,tanimotosim1,eucldist_vectorized1,cosresult1), axis=1)
te=pd.read_csv("chusaitezheng.csv")
te2= pd.concat((te,Tfidf12),axis=1)
te2.to_csv("te2.csv",index=None)
from sklearn import model_selection as ms
X_train,X_test,y_train,y_test = ms.train_test_split(te2,data[4],test_size=0.1,random_state=10)
#te.to_csv("chusaitezheng.csv")



from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
clf.fit(X_train, y_train)
test_pro = clf.predict_proba(X_test)
test_pro1=pd.DataFrame(test_pro)


from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,test_pro1.iloc[:,1] )#验证集上的auc值
print( test_auc )

##================================================================================
import sys
#print(sys.path)
sys.path.append("d:\\anaconda3-5.2.0\\lib\\site-packages")
import lightgbm as lgb  
import pandas as pd  
import numpy as np  
import pickle  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split  

  # create dataset for lightgbm  
lgb_train = lgb.Dataset(X_train, y_train)  
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
# specify your configurations as a dict  
params = {  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'binary_logloss', 'auc'},  
    'num_leaves': 5,  
    'max_depth': 6,  
    'min_data_in_leaf': 450,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,  
    'bagging_fraction': 0.95,  
    'bagging_freq': 5,  
    'lambda_l1': 1,    
    'lambda_l2': 0.001,  # 越小l2正则程度越高  
    'min_gain_to_split': 0.2,  
    'verbose': 5,  
    'is_unbalance': True  
}  
  
# train  
print('Start training...')  
gbm = lgb.train(params,  
                lgb_train,  
                num_boost_round=10000,  
                valid_sets=lgb_eval,  
                early_stopping_rounds=500)  
  
print('Start predicting...')  
preds = pd.DataFrame(gbm.predict(X_test, num_iteration=gbm.best_iteration))  # 输出的是概率结果  
from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,preds)#验证集上的auc值
print( test_auc )





