#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 8:29
# @Author  : YuanJing
# @File    : imputation16273.py
"""
Description:
    本程序的主要功能为处理从数据库中提取出的原始数据，得到用于机器学习模型的特征值矩阵
    处理流程为：
        1.去掉建模无关的项：病人编号、医院编号、入院时间、ICU编号
        2.去掉不能用于建模的结局变量：出院死亡率、ICU停留时间
        3.处理身高、体重、年龄、性别：
            身高、体重生成组合变量BMI、年龄大于200岁的用年龄的中位数91.4代替、性别F用1表示
        4.处理缺失数据：筛选出缺失比例在40%以内的特征值，对缺失数据用平均值进行插补
        5.处理异常数据：首先判断数据是否服从正态分布，不符合可能需要进行取对数处理，使其服从正态分布。
            本程序中的数据服从正态分布，因此直接用改进z_score 及Tukery方法进行异常值识别，两种方法皆识别
            为异常值的认为是异常值；将异常值用平均值代替。
Input:
    'final16273.csv'：数据库提取结果，16273表示前后都用6小时平均尿量少于0.5ml/kg/h标准的提取结果
    ‘final25789.csv'：数据库提取结果，25789表示用24小时尿量小于400ml定义少尿的提取结果
Output:
    'outlier16273.csv'：少尿标准用6小时定义的数据处理结果，用于机器学习算法
    ’outlier25789.csv'：少尿标准用24小时定义的数据处理结果，用于机器学习算法
"""
import numpy as np
import  pandas as pd
import OliguriaFunction as OF
import scipy.stats as statest
from sklearn.preprocessing import Imputer


#加载数据
data=pd.read_csv('./data/final16273.csv')

#数据预处理，去掉建模无关项
data.drop(['subject_id','hadm_id','intime','icustay_id'],inplace=True,axis=1)#去掉建模无关的列
data.drop(['hospmor'],inplace=True ,axis=1)#出院死亡率为结局变量，不能用来训练模型
data.drop(['icu_length_of_stay'],inplace=True ,axis=1)#去除ICU住院时长，结局变量，不能用来训练模型
data.drop(['si_max', 'si_mean', 'si_min', 'sirs','saps','diuretic'],inplace=True,axis=1)#去掉不合理的特征值si，利尿剂、以及两个不相关的评分

#将身高体重合成BMI，然后去掉身高、体重列
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

#提取标签列
labelmat=data['classlabel']
data.drop(['classlabel'],inplace=True ,axis=1)#去掉标签列

#统计缺失比例，保留缺失比例40%以内的特征值，并对缺失数据用平均值进行插补
nullpercent=data.count()/16273#统计每个特征值的缺失比例
feature40names = nullpercent[nullpercent >= 0.6].index#提取缺失比例小于40%的特征值名称
data = data[feature40names]#提取缺失比例小于40%的特征值数据
featurenames = data.keys()#能用的特征值名称
imp=Imputer(strategy='mean')#缺失数据用平均值进行插补
data=imp.fit_transform(data)

num_features=np.shape(data)[1]#特征值个数

#-----------------------------------------------------------------#
"""
检验数据是否服从正态分布
scipy.sats.normaltest
得到的p值都是0左右，都服从正态分布？？
"""
statistic, ptest = statest.normaltest(data,axis=0)#检验每一个特征是否服从正态分布
#-----------------------------------------------------------------#
data=OF.zscore_re(data)#去异常值
data=imp.fit_transform(data)#异常值用平均值插补
data=pd.DataFrame(data,columns=featurenames)
data['classlabel']=labelmat
data.to_csv('./data/outlier16273.csv')#最终用于机器学习模型中的完整数据集
# datamat=OF.normalizedata(data)
print()