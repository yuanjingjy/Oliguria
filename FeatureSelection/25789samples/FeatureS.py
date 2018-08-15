#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/29 9:47
# @Author  : YuanJing
# @File    : FeatureS.py

"""
输入信息：
    1.根据Gini_index、ReliefF、Fisher_score计算结果对特征值的排序文件，FSsort.csv
    2.原始数据文件final_4456.csv
"""

import  pandas as pd
import OliguriaFunction as OF
import numpy as np
import  ann
from sklearn.neural_network import MLPClassifier  # import the classifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import adaboost
import logRegres as LR
from sklearn import svm

sortinfo=pd.read_csv('.../data/FSsort.csv')
data=pd.read_csv('final_4456.csv')

sortname=sortinfo.ix[:,0]
datasorted=data[sortname]

#去除假设检验不通过的
delnames=['diuretic','sirs','meanbp_mean','heartrate_mean',
          'bicarbonate_max','rbc_min','pco2_avg','rbc_max',
          'rbc_avg','ph_avg','resprate_mean','BMI','resprate_min',
          'creatinine_min','bun_avg','inr_min','bun_min','bun_max',
          'creatinine_avg','spo2_mean','diasbp_max','pt_min',
          'platelet_avg','creatinine_max','spo2_max','wbc_min',
          'wbc_avg','wbc_max','platelet_max','pt_avg']
dataused=datasorted.drop(delnames,axis=1)

labelmat=data['classlabel']
names=dataused.keys()
# datamat=OF.normalizedata(dataused)
# datamat=pd.DataFrame(dataused,columns=names)
datamat=dataused
datamat['classlabel']=labelmat
datamat.to_csv('FeatureSorted.csv', encoding='utf-8',index=False)
n=np.shape(datamat)[1]


meanfit=[]#用来存储逐渐增加特征值过程中，不同数目特征值对应的BER平均值
stdfit=[]#用来存储逐渐增加特征值过程中，不同数目特征值对应的BER标准差
for i in range(3):
    print("第%s个参数："%(i+1))
    index = names[0:i + 1]
    dataMat = datamat.loc[:, index]
    dataMat=np.array(dataMat)
    labelMat=labelmat

    skf = StratifiedKFold(n_splits=10)
    scores=[]#用来存十折中每一折的BER得分
    mean_score=[]#第i个特征值交叉验证后BER的平均值
    std_score=[]#第i个特征值交叉验证后BER的标准差
    k=0;

    for train, test in skf.split(dataMat, labelMat):
        k=k+1
        # print("%s %s" % (train, test))
        print("----第%s次交叉验证：" %k)
        train_in = dataMat[train]
        test_in = dataMat[test]
        train_out = labelMat[train]
        test_out = labelMat[test]

#------------------------------------------------------------------------------------
        test_predict=OF.LRFeature(train_in,train_out,test_in)#此处用于更换不同的算法
#  ------------------------------------------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(test_out, test_predict).ravel()
        BER=0.5*((fn/(tp+fn))+(fp/(tn+fp)))
        scores.append(BER)
    mean_score = np.mean(scores)
    std_score=np.std(scores)

    meanfit.append(mean_score)
    stdfit.append(std_score)
#==============================================================================

meanfit = np.array(meanfit)
writemean=pd.DataFrame(meanfit)
writemean.to_csv('LRmean.csv', encoding='utf-8', index=True)


stdfit=np.array(stdfit)
writestd=pd.DataFrame(stdfit)
writestd.to_csv('LRfit.csv', encoding='utf-8', index=True)

fig, ax1 = plt.subplots()
line1 = ax1.plot(meanfit, "b-", label="BER")
ax1.set_xlabel("Number of features")
ax1.set_ylabel("BER", color="b")
plt.show()


print("test")





