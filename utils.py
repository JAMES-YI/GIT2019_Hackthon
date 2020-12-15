# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:59:53 2019

References
[1] https://thedatafrog.com/text-mining-pandas-yelp/
[2] https://stackoverflow.com/questions/9233027/unicodedecodeerror-charmap-codec-cant-decode-byte-x-in-position-y-character
[3] https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups/38309823
[4] https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
[5] https://stackoverflow.com/questions/25646200/python-convert-timedelta-to-int-in-a-dataframe

@author: jiryi
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.linear_model as sklm
from sklearn.ensemble import RandomForestRegressor
# open input file: 

def LoadJsonData(FilePath,Attributes,TotNum=None):
    '''
    Load json data. 
    - FilePath: path to the json file
    - Attributes: a list, which attributes should be loaded
    
    attributes of yelp_academic_dataset_business.json:
        ['business_id','name','address', 'city','state','postal_code',
        'latitude','longitude','stars','review_count','is_open','attributes',
        'categories','hours'] 
        => [business_id,latitude,longitude,stars,review_count,
            take_out,business_parking,
            categories_number,average_hours_per_day]
    attributes of yelp_academic_dataset_checkin.json:
        ['business_id','date']
    attributes of yelp_academic_dataset_review.json:
        ['review_id','user_id','business_id','stars','date',
        'text','useful','funny','cool']
        totally 6,685,900; 1637138 users, 192606 business;
        => stars
    attributes of yelp_academic_dataset_tip.json:
        ['text','date','business_id','user_id']
    attributes of yelp_academic_dataset_user.json:
        ['user_id','name','review_count','yelping_since',
        'friends','useful','funny','cool','fans','elite',
        'average_stars','compliment_hot','compliment_more',
        'compliment_profile','compliment_cute','compliment_list',
        'compliment_note','compliment_plain','compliment_cool',
        'compliment_funny','compliment_writer','compliment_photos']
        
    '''
    
    
    DataList = list()
    with open(FilePath,encoding='utf8') as OFile: 
        for i, line in enumerate(OFile):
            # print(f"(i,line): {i,line}\n") 
            if TotNum != None and i==TotNum:
                print(f'line: {line}\n')
                break
            if i == 10000:
                print(f"(i,line): {i,line}\n")
            
            DataTemp = json.loads(line)
            AttValTemp = list()
            for iAtt in Attributes:
                AttValTemp.append(DataTemp[iAtt])
            
            DataList.append(AttValTemp)
        
    DataFr = pd.DataFrame(DataList, columns=Attributes)
    # print(DataFr.head(5))
    
    return DataFr

def GenRevStaMat():
    '''
    generate comment matrix
    '''
    FilePath = 'yelp_academic_dataset_review.json'
    Attributes = ['user_id','business_id','stars']
    UseBusSta = LoadJsonData(FilePath,Attributes,TotNum=None)
    return UseBusSta
    

def Empty():
    
    
    return 0

def DataExp(DF):
    
    for ColInd,ColName in enumerate(DF.columns):
        # print(DF)
        print(f"{DF.columns[ColInd]}:\n {DF[DF.columns[ColInd]][0:3]}\n")

def TimePrepro(TS):
    
    FMT = '%Y-%m-%d %H:%M:%S'
    CurrT = dt.now().strftime(FMT)
    
    for Ind,Val in enumerate(TS):
        TSVal = dt.strptime(CurrT,FMT) - dt.strptime(Val,FMT)
        TS[Ind] = TSVal.days
    
def TimePreproNormal(TS):
    
    return 0

def CountPrepro(DS):
    
    for Ind, Val in enumerate(DS):
        
        if Val=='':
            DS[Ind] = 0
        else:
            DS[Ind] = Val.count(',') + 1
    
    return DS

def CountPreproNormal(DS):
    
    return 0

def CorlCheck(DF):
    
    plt.figure(figsize=(10,10))
    CorlCoeff = DF.corr()
    print(f'CorlCoeff:\n {CorlCoeff}')
    sns.heatmap(CorlCoeff,annot=True)
    plt.show()
    
    return 0

if __name__ == "__main__":
    
#    FilePath = 'yelp_academic_dataset_business.json'
#    Attributes = ['business_id','name','address',
#                         'city','state','postal_code',
#                         'latitude','longitude','stars',
#                         'review_count','is_open','attributes',
#                         'categories','hours']
    
#    FilePath = 'yelp_academic_dataset_review.json'
#    Attributes = ['review_id','user_id','business_id','stars','date',
#                  'text','useful','funny','cool']
    FilePath = 'yelp_academic_dataset_user.json'
    Attributes= ['user_id','yelping_since','friends','review_count','elite',
                 'average_stars','useful','funny','cool','fans']
    DF = LoadJsonData(FilePath,Attributes,TotNum=1000) # Take some time to load; suggest to 

#    UserDataOriPath = 'user_ori.csv'
#    UserDataNumePath = 'user_nume.csv'
    # DF.to_csv(r'user_ori.csv',header=True)
    # DFNume.to_csv(r'user_nume.csv',header=True)
    '''
    All users are unique, no duplications, 1637138
    '''
    DataExp(DF)
    
#    DF.yelping_since.apply(TimePrepro)
    TS = TimePrepro(DF.yelping_since) 
    FS = CountPrepro(DF.friends)
    ES = CountPrepro(DF.elite)
    DFNume = DF[['yelping_since','friends','review_count','elite',
                 'average_stars','useful','funny','cool','fans']].astype('int64')
    
    CorlCheck(DFNume)
    DFNume = DF[['yelping_since','friends','review_count','elite',
                 'average_stars','useful','fans']]
    

    Train,Test = train_test_split(DFNume,test_size=0.2)
    TrainFeat = Train[['yelping_since','friends','review_count',
                      'elite','average_stars','useful']]
    TrainLab = Train['fans']
    TestFeat = Test[['yelping_since','friends','review_count',
                      'elite','average_stars','useful']]
    TestLab = Test['fans']
    
    
    
    
    LinMod = sklm.LinearRegression()
    CVNum = KFold(n_splits=5,shuffle=True,random_state=42)
    LinModVal = LinMod.fit(TrainFeat,TrainLab)
    LinMod.score(TestFeat,TestLab)
    LinMod.score(TrainFeat,TrainLab)
    
    from sklearn.ensemble import RandomForestRegressor
    
#    for i,(TFVal,TLVal) in enumerate(CVNum.split(TrainFeat,TrainLab)):
#        
#        LinMod.fit(TrainFeat.iloc[TFVal,:],TrainLab.iloc[TLVal,:])
#    
    Rdf = RandomForestRegressor(n_estimators=10)
    Rdf.fit(TrainFeat,TrainLab)
    Rdf.score(TrainFeat,TrainLab)
    Rdf.score(TestFeat,TestLab)
    Rdf.feature_importances_
    Rdf.predict([[0,1,2,3,4,5],[98,89,29,38,19,2]])
    
    
    
    
    
            
#    
#    
#    DataFr = LoadJsonData(FilePath,Attributes)
#    RecMat = GenRevStaMat()
#    BusCount = RecMat.business_id.value_counts()
#    UserCount = RecMat.user_id.value_counts()
#    BusCount.describe([0.1,0.3,0.5,0.7,0.9,0.95,0.97])
#    UserCount.describe([0.1,0.3,0.5,0.7,0.9,0.95,0.97])
    
    
    
    
    

