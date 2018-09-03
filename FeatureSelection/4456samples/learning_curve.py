#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 16:49
# @Author  : YuanJing
# @File    : tmp1.py

# 学习曲线诊断偏差和方差问题
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris
import xgboost as xg

import pandas as  pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  AdaBoostClassifier

data = pd.read_csv('./data/outlier4456.csv')
labelMat = data['classlabel']
dataMat=data.drop('classlabel',axis=1)

# 去除假设检验不通过的
# 去除假设检验不通过的
delnames = [ 'pco2_avg', 'ph_avg','wbc_min','wbc_max','wbc_avg',
             'rbc_min','rbc_max','rbc_avg','platelet_max','platelet_avg',
             'creatinine_min','creatinine_avg','bun_min',
             'bun_max','bun_avg','pt_min','pt_avg','inr_min',
             'heartrate_min','diasbp_max','meanbp_mean',
             'resprate_mean','resprate_min','spo2_max',
             'spo2_min','spo2_mean','BMI']#16273个样本假设检验不通过的
dataMat = dataMat.drop(delnames, axis=1)

clf = MLPClassifier(hidden_layer_sizes=(45,),
                        activation='relu',
                        shuffle=False,
                        solver='sgd',
                        alpha=1e-6,
                        batch_size=5,
                        early_stopping=True,
                        max_iter=10000
                        # ,learning_rate='adaptive'
)
# clf =AdaBoostClassifier( n_estimators=50,algorithm='SAMME.R',learning_rate=0.5)#8,9
X_train = dataMat
y_train = labelMat

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', clf)])

train_sizes, train_scores, test_scores = \
    learning_curve(estimator=pipe_lr,
                   X=X_train,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10,
                   n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 0.8])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()
