#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 14:40
# @Author  : YuanJing
# @File    : tmp.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.pipeline import  Pipeline
from  sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

data = pd.read_csv('./data/outlier4456.csv')
labelMat = data['classlabel']
dataMat=data.drop('classlabel',axis=1)

# 去除假设检验不通过的
delnames = [ 'po2_avg', 'pco2_max', 'ph_min','rbc_max',
             'creatinine_min','creatinine_max','creatinine_avg',
             'bun_max','bun_avg','pt_min','inr_min','meanbp_mean',
             'resprate_mean','resprate_min','spo2_max',
             'spo2_min']#16273个样本假设检验不通过的
dataMat = dataMat.drop(delnames, axis=1)

dataMat = StandardScaler().fit_transform(dataMat)

# param_range = np.logspace(-6, -1, 5)
param_range = np.linspace(1, 45, 10)
param_range = param_range.astype('int')
train_scores, test_scores = validation_curve(
    estimator = MLPClassifier(
                        activation='relu',
                        shuffle=True,
                        solver='adam',
                        alpha=1e-6,
                        batch_size=5,
                        early_stopping=True,
                        max_iter=10000
                        # ,learning_rate='adaptive'
                        ),
    X=dataMat,
    y=labelMat,
    param_name='hidden_layer_sizes',
    param_range=param_range,
    cv=10,
    scoring="accuracy",
    n_jobs=1)


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
# plt.semilogx(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
plt.plot(param_range,train_scores_mean,color="darkorange")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
plt.plot(param_range,test_scores_mean,color="darkorange")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()