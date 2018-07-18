#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 8:29
# @Author  : YuanJing
# @File    : imputation16273.py

import numpy as np
import  pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import OliguriaFunction as OF
import scipy.stats as statest
import  math
import matplotlib.pyplot as plt


#加载数据
data=pd.read_csv('final16273.csv')

#数据预处理，去掉建模无关项
data.drop(['subject_id','hadm_id','intime','icustay_id'],inplace=True,axis=1)#去掉建模无关的列
data.drop(['hospmor'],inplace=True ,axis=1)#出院死亡率为结局变量，不能用来训练模型
data.drop(['icu_length_of_stay'],inplace=True ,axis=1)#去除ICU住院时长，结局变量，不能用来训练模型
BMI=10000*data.weight/(data.height*data.height)
data['BMI']=BMI
data.drop(['weight'],inplace=True ,axis=1)#用BMI
data.drop(['height'],inplace=True ,axis=1)#用BMI

[n_sample,n_feature]=np.shape(data)

#处理年龄、性别变量
for i in range(n_sample):
    if data.ix[i,'gender']=='F':
        data.ix[i,'gender']=1#女性用1表示
    else:
        data.ix[i,'gender']=0#男性用0表示

    if data.ix[i,'age']>200:
        data.ix[i,'age']=91.4#年龄300的用中位数91.4代替

#data.to_csv('final_4456.csv')

labelmat=data['classlabel']
data.drop(['classlabel'],inplace=True ,axis=1)#去掉标签列

data=data.drop(['vaso','saps','sapsii','sapsii_prob','lods',
                      'oasis','oasis_prob','mingcs','apsiii_prob','apsiii'
                      ,'vent','sofa','diuretic','sirs'] ,axis=1)
nullpercent=data.count()/16273
nullpercent = pd.DataFrame(nullpercent)
feature40=nullpercent  >= 0.4
feature40names[feature40.keys()
data777=data[feature40names]

datamat=data
featurenames=datamat.keys()
num_features=np.shape(data)[1]

#-----------------------------------------------------------------#
"""
检验数据是否服从正态分布
scipy.sats.normaltest
得到的p值都是0左右，都服从正态分布？？
"""
pvalue=[]
kvalue=[]
JBtest=[]
statistic, ptest = statest.normaltest(data,axis=0)#检验每一个特征是否服从正态分布

#-----------------------------------------------------------------#
data=OF.zscore_re(data)
data.fillna(data.mean(),inplace=True)
data['classlabel']=labelmat
data.to_csv('异常处理后.csv')
datamat=OF.normalizedata(data)