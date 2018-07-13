#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 15:12
# @Author  : YuanJing
# @File    : hyperpar.py
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import OliguriaFunction as OF
from sklearn.linear_model import LogisticRegression

###################################################

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


def percept(args):
    global dataMat,labelMat
    accuracy=[]
    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(dataMat, labelMat):
        print("%s %s" % (train, test))
        train_in = dataMat[train]
        test_in = dataMat[test]
        train_out = labelMat[train]
        test_out = labelMat[test]

        clf = LogisticRegression(penalty=args["penalty"], C=args["C"],
                                intercept_scaling=args["intercept_scaling"],
                                 class_weight='balanced',
                                 solver='liblinear', max_iter=1000,
                                  verbose=args["verbose"],
                                 warm_start=args["warm_start"])
        clf.fit(train_in,train_out)
        y_pred=clf.predict(test_in)
        acc=accuracy_score(test_out,y_pred)
        accuracy.append(acc)
    return -np.mean(accuracy)


from hyperopt import fmin,tpe,hp,partial
space={"penalty":hp.choice("penalty",['l1','l2']),
       "C":hp.uniform("C",0.01,50),
       "intercept_scaling":hp.uniform("intercept_scaling",1,50),

       "warm_start":hp.choice("warm_start",['True','False']),
       "verbose":hp.randint("verbose",20),
       }
algo=partial(tpe.suggest)
best=fmin(percept,space,algo=algo,max_evals=100)
print(best)
print(percept(best))
print("test")

