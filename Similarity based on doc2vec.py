# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:02:28 2019

@author: Administrator
"""

import sys
#print(sys.path)
sys.path.append("c:\\users\\administrator\\appdata\\local\\programs\\python\\python37-32\\lib\\site-packages\\")
from gensim.models import Doc2Vec
import numpy
import logging
import gensim
import pandas as pd
import numpy as np
data=pd.read_csv("train_data.csv",header=-1)
data2 = pd.concat((data[1],data[3]), axis=1)
data3=np.array(data2)
data3=data3.tolist()
help(Doc2Vec)
aggededDocument = gensim.models.doc2vec.TaggedDocument
model = Doc2Vec(data3, size=300, hs=1, min_count=1, window=3)
ws1=np.array(data2[1])
ws2=np.array(data2[3])
ws1=ws1.tolist()
ws2=ws2.tolist()
s2=pd.DataFrame()
for i in range(len(data)):
    ws3=[ws1[i]]
    ws4=[ws2[i]]
    s1=model.wv.n_similarity(ws3,ws4)
    s3=pd.DataFrame(pd.Series(s1))
    s2=s2.append(s3)
s2=s2.reset_index(drop=True)
from sklearn import model_selection as ms
X_train,X_test,y_train,y_test = ms.train_test_split(s2,data[4],test_size=0.1,random_state=10)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
clf.fit(X_train, y_train)
test_pro = clf.predict_proba(X_test)
test_pro1=pd.DataFrame(test_pro)
from sklearn import metrics
test_auc = metrics.roc_auc_score(y_test,test_pro1.iloc[:,1] )#验证集上的auc值
print( test_auc )