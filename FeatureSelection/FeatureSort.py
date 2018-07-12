#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 15:11
# @Author  : YuanJing
# @File    : FeatureSort.py

"""
Description：
    1.首先计算Relief、Fisher-score、Gini_index三个得分值，归一化后叠加到一起得到最终分值
    2.根据叠加后的分值对特征值进行排序

OutputMessage:
    sorteigen：记录的是各个方法得到的特征值评分以及根据整合后评分进行排序后的结果，
                        对应FSsort.csv
"""
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
data=pd.read_csv('addsex4456.csv')

#数据预处理，去掉建模无关项
data.drop(['subject_id','hadm_id','intime','icustay_id'],inplace=True,axis=1)#去掉建模无关的列
data.drop(['hospmor'],inplace=True ,axis=1)#出院死亡率为结局变量，不能用来训练模型
data.drop(['icu_length_of_stay'],inplace=True ,axis=1)#去除ICU住院时长，结局变量，不能用来训练模型
BMI=10000*data.weight/(data.height*data.height)
data['BMI']=BMI
data.drop(['weight'],inplace=True ,axis=1)#用BMI
data.drop(['height'],inplace=True ,axis=1)#用BMI

[n_sample,n_feature]=np.shape(data)         
n_feature=n_feature-1

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
                      'oasis','oasis_prob','mingcs','apsiii_prob','apsiii',
                      'gender','vent','sofa','diuretic','sirs'] ,axis=1)

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

#------------calculate the FS score with scikit-feature package--------------#
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import gini_index

Relief = reliefF.reliefF(datamat, labelmat)
Fisher= fisher_score.fisher_score(datamat, labelmat)
gini= gini_index.gini_index(datamat,labelmat)
gini=-gini
FSscore=np.column_stack((Relief,Fisher,gini))#合并三个分数

FSscore=OF.normalizedata(FSscore)
FinalScore=np.sum(FSscore,axis=1)
FS=np.column_stack((FSscore,FinalScore))
FS_nor=OF.normalizedata(FS)#将最后一列联合得分归一化
FS=pd.DataFrame(FS_nor,columns=["Relief", "Fisher","gini","FinalScore"],index=featurenames)
FS.to_csv("F:\F盘\Project\datathon少尿\Oliguria\Oliguria\FeatureSelection\FSscore_out.csv")

sorteigen=FS.sort_values(by='FinalScore',ascending=False,axis=0)
sorteigen.to_csv('FSsort_out.csv')

print("test")
