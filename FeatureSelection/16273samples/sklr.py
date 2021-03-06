#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 8:32
# @Author  : YuanJing
# @File    : sklr.py

# -*- coding: utf-8 -*-
"""
Description:
    机器学习算法进行少尿预警的主程序

"""

import ann
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import  LogisticRegression
import OliguriaFunction as OF
from sklearn.neural_network import MLPClassifier  # import the classifier
import pandas as  pd  # python data analysis
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import  AdaBoostClassifier
import xgboost as xg
from sklearn import svm
import seaborn as sns


data = pd.read_csv('./data/outlier16273.csv')
labelMat = data['classlabel']
dataMat=data.drop('classlabel',axis=1)

# 去除假设检验不通过的
delnames = [ 'po2_avg', 'pco2_max', 'ph_min','rbc_max',
             'creatinine_min','creatinine_max','creatinine_avg',
             'bun_max','bun_avg','pt_min','inr_min','meanbp_mean',
             'resprate_mean','resprate_min','spo2_max',
             'spo2_min']#16273个样本假设检验不通过的
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
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    train_in,train_out = RandomOverSampler().fit_sample(train_in,train_out)
    #
    # clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=5.7,#LR 10,11
    #                        fit_intercept=True, intercept_scaling=97, class_weight='balanced',
    #                        random_state=None, solver='liblinear', max_iter=10000,
    #                        multi_class='ovr',  warm_start=True)
    clf = MLPClassifier(hidden_layer_sizes=(45,),
                        activation='relu',
                        shuffle=True,
                        solver='sgd',
                        alpha=1e-6,
                        batch_size=5,
                        early_stopping=True,
                        max_iter=10000
                        # ,learning_rate='adaptive'
                        )
    # clf=xg.XGBClassifier()
    # clf = svm.SVC(C=0.1, kernel='rbf', gamma='auto',
    #               shrinking=True, probability=True, tol=0.0001,
    #               cache_size=1000, max_iter=-1, class_weight='balanced',
    #               decision_function_shape='ovr', random_state=None
    #               )
    # clf=AdaBoostClassifier( n_estimators=150,algorithm='SAMME.R',learning_rate=0.8)#8,9
    # clf=BaggingClassifier(n_estimators=200,max_samples=1.0,max_features=1.0,
    #                       bootstrap=True,bootstrap_features=False,random_state=200)
    clf.fit(train_in,train_out)
    test_predict = clf.predict(test_in)
    proba_test = clf.predict_proba(test_in)

    train_predict = clf.predict(train_in)
    proba_train = clf.predict_proba(train_in)

    test1,test2 = ann.evaluatemodel(train_out,train_predict,proba_train[:,1])
    evaluate_train.extend(test1)
    prenum_train.extend(test2)

    test3, test4 = ann.evaluatemodel(test_out, test_predict, proba_test[:,1])  # test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
# Result_test.to_csv('BER_LR_ks.csv')
Result_test.boxplot()
plt.show()

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)

evaluate_test = np.array(evaluate_test)
prenum_test = np.array(prenum_test)


mean_train = np.mean(evaluate_train,axis=0)
std_train = np.std(evaluate_train,axis=0)
evaluate_train.append(mean_train)
evaluate_train.append(std_train)

evaluate_train = np.array(evaluate_train)
prenum_train = np.array(prenum_train)

print(evaluate_test)