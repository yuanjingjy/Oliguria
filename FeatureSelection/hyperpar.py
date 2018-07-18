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
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier#import the classifier
from sklearn import svm

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
n_samples,n_features=np.shape(dataMat)

def percept(args):
    global dataMat,labelMat
    accuracy=[]
    skf = StratifiedKFold(n_splits=5)
    for train, test in skf.split(dataMat, labelMat):
        # print("%s %s" % (train, test))
        train_in = dataMat[train]
        test_in = dataMat[test]
        train_out = labelMat[train]
        test_out = labelMat[test]
        train_in, train_out = RandomOverSampler().fit_sample(train_in, train_out)

        # clf = LogisticRegression(penalty='l2', C=args["C"],
        #                         intercept_scaling=args["intercept_scaling"],
        #                          class_weight='blanced',
        #                          solver=args['solver'], max_iter=1000,
        #                           verbose=args["verbose"],
        #                          warm_start=True)
        clf = MLPClassifier(hidden_layer_sizes=(int(args['hidden_layer_sizes']),),
                            activation=args['activation'],
                            shuffle=True,
                            solver=args['solver'],
                            alpha=1e-6,
                            batch_size=int(args['batch_size']),
                            early_stopping=args['early_stopping'],
                            max_iter=1000
                            )
        # clf = svm.SVC(C=args['C'],kernel=args['kernel'], gamma='auto',
        #               shrinking=True,  probability=True,  tol=0.0001,
        #               cache_size=1000,  max_iter=-1, class_weight='balanced',
        #               decision_function_shape='ovr', random_state=None
        #              )

        clf.fit(train_in, train_out)
        y_pred = clf.predict(test_in)
        acc=accuracy_score(test_out, y_pred)
        accuracy.append(acc)
    return -np.mean(accuracy)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,partial,space_eval
space={
    'hidden_layer_sizes': hp.uniform('hidden_layer_sizes', 2, n_features),
    'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
    'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
    'batch_size': hp.uniform('batch_size', 1, 100),
    'early_stopping': hp.choice('early_stopping', [True, False]),
       }
space_SVM={
    'C':hp.uniform('C',0.1,50),
    'kernel':hp.choice('kernel',['linear','poly','rbf','sigmoid','precomputed']),
    'degree':hp.uniform('degree',1,10),
    'coef0':hp.uniform('coef0',0,10),
}
algo=partial(tpe.suggest)
trials = Trials()
best=fmin(percept,space,algo=algo,max_evals=100,trials=trials)
# print(best)
print(space_eval(space, best))
print(percept(space_eval(space, best)))
print("test")

