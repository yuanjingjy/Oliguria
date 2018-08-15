#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 8:32
# @Author  : YuanJing
# @File    : sklr.py

# -*- coding: utf-8 -*-
"""
Description:
    机器学习算法进行少尿预警的主程序，针对的是用24小时尿量定义的少尿
    不平衡数据的特殊处理：a.十折交叉验证的每一折中，都将阴性样本分成15份，分别与阳性样本构成平衡的训练数据
    b.用15个平衡后的数据集训练15个机器学习模型，分别对该折的测试集数据进行预测c.测试集的最终预测标签通过对15个模型
    的预测结果进行投票得到，预测概率为所有预测为该类的模型的概率的平均值

"""

import ann
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import OliguriaFunction as OF
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # import the classifier
from sklearn import  svm
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as  pd  # python data analysis
import matplotlib.pyplot as plt
import  xgboost as xg

data = pd.read_csv('./data/outlier25789.csv')
labelMat = data['classlabel']
dataMat=data.drop('classlabel',axis=1)

# 去除假设检验不通过的
delnames = [ 'wbc_min', 'wbc_avg', 'wbc_max', 'ph_avg', 'ph_min','ph_max',
             'platelet_max','lactate_max','lactate_avg','sirs','saps',
             'diasbp_max','sysbp_max','meanbp_max','mingcs','BMI']#25789个样本假设检验不通过的
# delnames = [ 'po2_avg', 'pco2_max', 'ph_min', 'ph_avg', 'ph_min','rbc_max',
#              'creatinine_min','creatinine_max','creatinine_avg','sirs','saps',
#              'bun_max','bun_avg','pt_min','inr_min','meanbp_mean',
#              'resprate_mean','resprate_min','spo2_max',
#              'spo2_min','si_max','diuretic']#16273个样本假设检验不通过的
dataMat = dataMat.drop(delnames, axis=1)
featurenames=dataMat.keys()

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test = []

dataMat = OF.normalizedata(dataMat)

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test = []

dataMat=np.array(dataMat)
labelMat = np.array(labelMat)
num_feantures = np.shape(dataMat)[1]
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(dataMat, labelMat):
    # ==============================================================================
    # skf=StratifiedShuffleSplit(n_splits=10)
    # for train,test in skf.split(dataMat,labelMat):
    # ==============================================================================
    print("%s %s" % (train, test))
    predict_ensemble = []#存放15个模型对测试集的预测结果
    predict_prob = []#存放15个模型预测出的概率值
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    #
    # clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=5.7,#LR 10,11
    #                        fit_intercept=True, intercept_scaling=97, class_weight='balanced',
    #                        random_state=None, solver='liblinear', max_iter=10000,
    #                        multi_class='ovr',  warm_start=True)
    clf = MLPClassifier(hidden_layer_sizes=(50,),#9,10
                        activation='tanh',
                        shuffle=True,
                        solver='sgd',
                        alpha=1e-6,
                        batch_size=5,
                        early_stopping=True,
                        max_iter=10000
                        # ,learning_rate='adaptive'
                        )
    # clf=xg.XGBClassifier()
    # clf = svm.SVC(C=45, kernel='rbf', gamma='auto',
    #               shrinking=True, probability=True, tol=0.0001,
    #               cache_size=1000, max_iter=-1, class_weight='balanced',
    #               decision_function_shape='ovr', random_state=None
    #               )
    # clf=AdaBoostClassifier( n_estimators=150,algorithm='SAMME.R',learning_rate=0.8)#8,9
    # clf=BaggingClassifier(n_estimators=200,max_samples=1.0,max_features=1.0,
    #                       bootstrap=True,bootstrap_features=False,random_state=200)
    trainset = np.c_[train_in, train_out]
    trainset=np.random.permutation(trainset)
    train_pos = trainset[trainset[:,num_feantures] == 1]#训练集中的全部阳性样本
    train_neg = trainset[trainset[:, num_feantures] == 0]#训练集中的全部阴性样本
    num_neg = np.shape(train_neg)[0]#阴性样本个数
    num_pos = np.shape(train_pos)[0]#阳性样本个数
    n_split=int(num_neg/num_pos)#划分多少个模型

    for i in range(15):#阴性样本分成15份，分别与相同的阳性样本构成平衡的训练集，训练15个模型
        print(i)
        if i < 14:
            neg = train_neg[(i*num_pos+1):((i+1)*num_pos),:]
        else:
            neg = train_neg[(i*num_pos+1):num_neg,:]
        train_i = np.r_[neg,train_pos]
        train_i = np.random.permutation(train_i)
        clf.fit(train_i[:,0:num_feantures],train_i[:,num_feantures])
        pre_i=clf.predict(test_in[:,0:num_feantures])
        pre_pro_i=clf.predict_proba(test_in)
        predict_ensemble.append(pre_i)
        predict_prob.append(pre_pro_i[:, 1])
    predict_prob = np.array(predict_prob)
    predict_ensemble = np.array(predict_ensemble)
    # print()

    sum_pre = np.sum(predict_ensemble,axis=0)
    tmp=sum_pre.copy()
    sum_pre[tmp >9] = 1
    sum_pre[tmp <10] = 0
    test_predict = sum_pre#测试集的预测结果

    sum_prob = []
    for j in range(np.shape(test_in)[0]):
        colj = predict_prob[:, j]
        if sum_pre[j] == 1:
            tmp_prob = np.mean(colj[colj >= 0.5])
        else:
            tmp_prob = np.mean(colj[colj < 0.5])
        # if tmp_prob == np.nan:
        #     print('test')
        sum_prob.append(tmp_prob)
    sum_prob = np.array(sum_prob)
    proba_test = sum_prob#测试集概率预测结果

    test3, test4 = ann.evaluatemodel(test_out, test_predict, proba_test)  # test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
Result_test.to_csv('BER_LR_ks.csv')
Result_test.boxplot()
plt.show()

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)


evaluate_test = np.array(evaluate_test)
prenum_test = np.array(prenum_test)


print(evaluate_test)