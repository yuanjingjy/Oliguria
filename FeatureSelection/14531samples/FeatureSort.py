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
data=pd.read_csv('./data/outlier14531.csv')
labelmat=data['classlabel']
data.drop(['classlabel'],inplace=True ,axis=1)#去掉标签列

datamat=data
featurenames=datamat.keys()
num_features=np.shape(data)[1]
data['classlabel']=labelmat

datamat = OF.standarddata(data)

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
# FS.to_csv(".\data\FSscore_out.csv")

sorteigen=FS.sort_values(by='FinalScore',ascending=False,axis=0)
sorteigen.to_csv('./data/FSsort_out.csv')

print("test")
