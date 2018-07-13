#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 8:32
# @Author  : YuanJing
# @File    : sklr.py

# -*- coding: utf-8 -*-
import ann
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import OliguriaFunction as OF
from sklearn.linear_model import LogisticRegression

import pandas as  pd  # python data analysis
import matplotlib.pyplot as plt

data = pd.read_csv('outlier.csv')
labelMat = data['classlabel']
dataMat = data.iloc[:, 0:60]
# dataMat=dataMat.drop(['vaso','saps','sapsii','sapsii_prob','lods',
#                       'oasis','oasis_prob','mingcs','apsiii_prob','apsiii',
#                       'gender','vent','sofa'] ,axis=1)

# 去除假设检验不通过的
delnames = ['pco2_avg', 'ph_avg', 'wbc_min', 'wbc_avg', 'wbc_max',
            'rbc_max', 'rbc_avg', 'rbc_min', 'ph_avg', 'platelet_avg', 'platelet_max',
            'creatinine_min', 'creatinine_avg', 'bun_min', 'bun_max',
            'bun_avg', 'pt_min', 'pt_avg', 'inr_min', 'heartrate_min', 'diasbp_max',
            'meanbp_mean', 'resprate_mean', 'resprate_min', 'spo2_mean',
            'spo2_max', 'spo2_min', 'BMI']
dataMat = dataMat.drop(delnames, axis=1)
featurenames=dataMat.keys()

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test = []

dataMat = OF.normalizedata(dataMat)
# addones = np.ones((4456, 1))
# dataMat = np.c_[addones, dataMat]

evaluate_train = []
evaluate_test = []
prenum_train = []
prenum_test = []

dataMat=np.array(dataMat)
labelMat = np.array(labelMat)

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
    train_in, train_out = RandomOverSampler().fit_sample(train_in, train_out)
    # train_in, train_out = RandomUnderSampler().fit_sample(train_in, train_out)

    clf=LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.3,
                           fit_intercept=True, intercept_scaling=1, class_weight='balanced',
                           random_state=None, solver='liblinear', max_iter=1000,
                           multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    clf.fit(train_in,train_out)
    test_predict=clf.predict(test_in)
    proba_test=clf.predict_proba(test_in)

    train_predict=clf.predict(train_in)
    proba_train=clf.predict_proba(train_in)


    len_train = np.shape(train_in)[0]
    len_test = np.shape(test_in)[0]
    test1, test2 = ann.evaluatemodel(train_out, train_predict, proba_train[:,1])  # test model with trainset
    evaluate_train.extend(test1)
    prenum_train.extend(test2)

    test3, test4 = ann.evaluatemodel(test_out, test_predict, proba_test[:,1])  # test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
Result_test.to_csv('BER_LR_ks.csv')
Result_test.boxplot()
plt.show()

mean_train = np.mean(evaluate_train, axis=0)
std_train = np.std(evaluate_train, axis=0)
evaluate_train.append(mean_train)
evaluate_train.append(std_train)

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)

evaluate_train = np.array(evaluate_train)
evaluate_test = np.array(evaluate_test)
prenum_train = np.array(prenum_train)
prenum_test = np.array(prenum_test)

evaluate_train_mean = np.mean(evaluate_test, axis=0)

print(evaluate_test)