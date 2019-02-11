# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
np.warnings.filterwarnings('ignore')



adult = pd.read_hdf('datasets.hdf','adult')
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

cancer = pd.read_hdf('cancer.hdf','cancer')        
cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values



adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.25, random_state=0,stratify=adultY)     
cancer_trgX, cancer_tstX, cancer_trgY, cancer_tstY = ms.train_test_split(cancerX, cancerY, test_size=0.25, random_state=0,stratify=cancerY)     



d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
d = cancerX.shape[1]
hiddens_cancer = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]


pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])  

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  



params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_cancer= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

cancer_clf = basicResults(pipeM,cancer_trgX,cancer_trgY,cancer_tstX,cancer_tstY,params_cancer,'KNN','cancer')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')        


adult_final_params={'KNN__n_neighbors': 160, 'KNN__p': 1, 'KNN__weights': 'uniform'}
adult_final_params=adult_clf.best_params_
cancer_final_params={'KNN__n_neighbors': 90, 'KNN__p': 1, 'KNN__weights': 'uniform'}
cancer_final_params=cancer_clf.best_params_



pipeM.set_params(**cancer_final_params)
makeTimingCurve(cancerX,cancerY,pipeA,'KNN','cancer')

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')



