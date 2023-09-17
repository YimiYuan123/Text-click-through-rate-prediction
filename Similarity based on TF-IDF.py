# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:25:46 2019

@author: MXB
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist
from sklearn import model_selection as ms

data=pd.read_csv("train_data.csv",header=-1)
y = data.iloc[:,4].values
vec = TfidfVectorizer()
Tfidf11= vec.fit_transform(data[3].values)
Tfidf12= pd.DataFrame(Tfidf11.todense())

cossim1= pd.DataFrame()
ChebyDistance1=pd.DataFrame()
ManhatDistance1=pd.DataFrame()
CorDistance1=pd.DataFrame()
eucldist_vectorized1=pd.DataFrame()
for i in range(len(data)):
    global datacos1
    global datacos2
    datacos1 = pd.Series(data.iloc[i,1])
    datacos2 = pd.Series(data.iloc[i,3])
    d = pd.concat([datacos1,datacos2],ignore_index=True)
    vec = TfidfVectorizer()
    Y = vec.fit_transform(d.values)
    m = Y.todense()
    Y1 = m[:1]
    Y2 = m[1:]
    X1 = np.vstack([Y1,Y2])
    cossim11 = pd.DataFrame(1 - pdist(X1,'cosine'))
    cossim1=cossim1.append(cossim11)
    Y3=np.array(Y1)
    Y4=np.array(Y2)
    ChebyDistance11=pd.DataFrame(pd.Series(np.max(np.abs( Y3-Y4))))
    ChebyDistance1=ChebyDistance1.append(ChebyDistance11)
    ManhatDistance11=pd.DataFrame(pd.Series(np.sum(np.abs(Y3-Y4))))
    ManhatDistance1=ManhatDistance1.append( ManhatDistance11)
    eucldist_vectorized11= np.sqrt(np.sum((Y3 - Y4)**2))
    eucldist_vectorized11= pd.DataFrame(pd.Series(eucldist_vectorized11))
    eucldist_vectorized1=eucldist_vectorized1.append(eucldist_vectorized11) 
    CorDistance11=pd.DataFrame(pd.Series(1-np.corrcoef(Y3,Y4)[0,1]))
    CorDistance1=CorDistance1.append(CorDistance11)
cossim1=cossim1.reset_index(drop=True)
cossim1=cossim1.fillna(0)
eucldist_vectorized1=eucldist_vectorized1.reset_index(drop=True)
ChebyDistance1= ChebyDistance1.reset_index(drop=True)
ManhatDistance1=ManhatDistance1.reset_index(drop=True)
CorDistance1=CorDistance1.reset_index(drop=True)
CorDistance1=CorDistance1.fillna(0)









