# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:04:33 2019

@author: jiryi
"""
import pandas as pd
import numpy as np
#import json
import matplotlib.pyplot as plt
#from datetime import datetime as dt
#import seaborn as sns
#import random 
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score

import sklearn.linear_model as sklm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def LoadData(DataSize):
    DF = pd.read_csv('user_featureslable_nonnormed.csv')
    if DataSize!=None:
#        DF = DF[random.sample(set(np.arange(len(DF))),DataSize)]
        DF = DF[0:DataSize]
    DFNume = DF[['review_count', 'average_stars', 'useful',
                 'friends_count', 'elite_count', 'time_length2', 'fans']]
    
    Train,Test = train_test_split(DFNume,test_size=0.2)
    TrainFeat = Train[['review_count', 'average_stars', 'useful',
                 'friends_count', 'elite_count', 'time_length2']]
    TrainLab = Train['fans']
    TestFeat = Test[['review_count', 'average_stars', 'useful',
                 'friends_count', 'elite_count', 'time_length2']]
    TestLab = Test['fans']
    
    return TrainFeat,TrainLab,TestFeat,TestLab

def LoadDataNM(DataSize):
    DF = pd.read_csv('user_featureslable_normed.csv')
    if DataSize!=None:
#        DF = DF[random.sample(set(np.arange(len(DF))),DataSize)]
        DF = DF[0:DataSize]
    DFNume = DF[['review_count_nm', 'average_stars_nm', 'useful_nm',
                 'friends_count_nm', 'elite_count_nm', 'time_length2_nm', 'fans']]
    
    Train,Test = train_test_split(DFNume,test_size=0.2)
    TrainFeat = Train[['review_count_nm', 'average_stars_nm', 'useful_nm',
                 'friends_count_nm', 'elite_count_nm', 'time_length2_nm']]
    TrainLab = Train['fans']
    TestFeat = Test[['review_count_nm', 'average_stars_nm', 'useful_nm',
                 'friends_count_nm', 'elite_count_nm', 'time_length2_nm']]
    TestLab = Test['fans']
    
    return TrainFeat,TrainLab,TestFeat,TestLab

# [] Linear regression model

def LinRegre(TrainFeat,TrainLab,TestFeat,TestLab):
    
    LinMod = sklm.LinearRegression()
    LinMod.fit(TrainFeat,TrainLab)
    TestAcc = LinMod.score(TestFeat,TestLab)
    TrainAcc = LinMod.score(TrainFeat,TrainLab)
#    CVNum = KFold(n_splits=5,shuffle=True)
#    for i,(TFVal,TLVal) in enumerate(CVNum.split(TrainFeat,TrainLab)):
#        
#        LinMod.fit(TrainFeat.iloc[TFVal,:],TrainLab.iloc[TLVal,:])
    
    return TrainAcc,TestAcc



# [] Quadratic regression model

def QuadRegre():
    
    return 0

 
# [] Random forest regression 
    
def RdfRegreCV(EstmNum,TrainFeat,TrainLab,TestFeat,TestLab):
    
    AccTrain = []
    AccTest = []
    for Ind,Val in enumerate(EstmNum):
        Rdf = RandomForestRegressor(Val) 
        Rdf.fit(TrainFeat,TrainLab)
        AccTrain.append(Rdf.score(TrainFeat,TrainLab))
        AccTest.append(Rdf.score(TestFeat,TestLab))
    
    return AccTrain, AccTest

def RdfRegre(EstmNum,TrainFeat,TrainLab,TestFeat,TestLab):
    
    Rdf = RandomForestRegressor(EstmNum) 
    Rdf.fit(TrainFeat,TrainLab)
    AccTrain = Rdf.score(TrainFeat,TrainLab)
    AccTest = Rdf.score(TestFeat,TestLab)
    
    return AccTrain, AccTest


def RdfCV(EstmNum):
    
    RdfTry = RandomForestRegressor() 
    RdfSch = RandomizedSearchCV(estimator = RdfTry, 
                                param_distributions = {"n_estimators": EstmNum}, 
                                cv = 2)
    RdfSch.fit(TrainFeat,TrainLab)
    print(f"Best params: {RdfSch.best_params_}\n")
    
    return 0

if __name__=="__main__":
    
    TrainFeat,TrainLab,TestFeat,TestLab = LoadDataNM(DataSize=None)
    
    EstmNumCV = [2,3,4,5,6,7,8,9,10]
    AccTrainCV, AccTestCV = RdfRegreCV(EstmNumCV,TrainFeat,TrainLab,TestFeat,TestLab)
    
    plt.figure()
    plt.plot(EstmNumCV,np.array(AccTrainCV),'-*')
    plt.plot(EstmNumCV,np.array(AccTestCV),'-o')
    plt.xlabel('# of estimators')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Testing'])
    plt.show()
    
    DataSizeArr = np.arange(1,12,1)*np.power(10,6)
    AccTrainArr = []
    AccTestArr = []
    AccTrainLinArr = []
    AccTestLinArr = []
    for DataSize in DataSizeArr:
        TrainFeat,TrainLab,TestFeat,TestLab = LoadDataNM(DataSize)
        
        AccTrain, AccTest = RdfRegre(10,TrainFeat,TrainLab,TestFeat,TestLab)
        AccTrainArr.append(AccTrain)
        AccTestArr.append(AccTest)
        
        AccTrainLin, AccTestLin = LinRegre(TrainFeat,TrainLab,TestFeat,TestLab)
        AccTrainLinArr.append(AccTrainLin)
        AccTestLinArr.append(AccTestLin)
        
    plt.figure()
    plt.plot(np.arange(1,12,1),np.array(AccTrainArr),'-*')
    plt.plot(np.arange(1,12,1),np.array(AccTestArr),'-o')
    plt.xlabel('Dataset size (x 10^6)')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Testing'])
    plt.show()
    
    plt.figure()
    plt.plot(np.arange(1,12,1),np.array(AccTrainLinArr),'-*')
    plt.plot(np.arange(1,12,1),np.array(AccTestLinArr),'-o')
    plt.xlabel('Dataset size (x 10^6)')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Testing'])
    plt.show()
    
    TrainFeat,TrainLab,TestFeat,TestLab = LoadDataNM(DataSize=None)
    Rdf = RandomForestRegressor(10) 
    Rdf.fit(TrainFeat,TrainLab)
    
    ImpSco = Rdf.feature_importances_
    Objs = ('review #', 'stars', 'useful',
            'friends #', 'elite #', 'time length')
    ObjsPos = np.arange(len(Objs))
    plt.figure()
    plt.bar(ObjsPos,ImpSco,alpha=0.5)
    plt.xticks(ObjsPos,Objs)
    plt.ylabel('Importance score')
    plt.show()
    
    
    
    


