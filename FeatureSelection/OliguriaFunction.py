#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/29 14:30
# @Author  : YuanJing
# @File    : OliguriaFunction.py

"""
定义本项目中经常用到的函数
"""

from sklearn import preprocessing
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import logRegres as LR
from sklearn.neural_network import MLPClassifier  # import the classifier
from sklearn import svm
import adaboost


"""
Description :
    按列归一化到[0,1]
Input:
    datain:原始数据
Output:
    scaledata:按列归一化到[0,1]之后的矩阵
"""
def normalizedata(datain):
    min_max_scaler = preprocessing.MinMaxScaler()
    scaledata = min_max_scaler.fit_transform(datain)
    return scaledata


"""
Description:
    针对LR算法
"""
def LRFeature(train_in,train_out,test_in):
    n_train = np.shape(train_in)[0]
    n_test = np.shape(test_in)[0]

    # ---------对于LR的特殊处理
    addones_train = np.ones((n_train, 1))
    train_in = np.c_[addones_train, train_in]  # 给训练集数据加1列1

    addones_test = np.ones((n_test, 1))
    test_in = np.c_[addones_test, test_in]  # 给测试集加一列1

    train_in, train_out = RandomOverSampler().fit_sample(train_in, train_out)

    trainWeights = LR.stocGradAscent1(train_in, train_out, 500)
    len_test = np.shape(test_in)[0]
    test_predict = []
    for i in range(len_test):
        test_predict_tmp = LR.classifyVector(test_in[i, :], trainWeights)
        test_predict.append(test_predict_tmp)
    test_predict = np.array(test_predict)
    return test_predict


"""
Description:
    逐个增加
"""
def ANNFeature(i,train_in,train_out,test_in):
    train_in, train_out = RandomOverSampler().fit_sample(train_in, train_out)

    clf = MLPClassifier(hidden_layer_sizes=(i + 1,), activation='tanh',
                        shuffle=True, solver='sgd', alpha=1e-6, batch_size=3,
                        learning_rate='adaptive')
    clf.fit(train_in, train_out)
    test_predict = clf.predict(test_in)
    return test_predict

"""
Description:
    逐个增加
"""
def SVMFeature(train_in,train_out,test_in):
    train_in, train_out = RandomOverSampler().fit_sample(train_in, train_out)
    clf = svm.SVC(C=50, kernel='rbf', gamma='auto', shrinking=True, probability=True,
                  tol=0.001, cache_size=1000, verbose=False,
                  max_iter=-1, decision_function_shape='ovr', random_state=None)
    clf.fit(train_in, train_out)  # train the classifier
    test_predict = clf.predict(test_in)  # test the model with trainset
    return test_predict

"""

"""
def AdaFeature(train_in,train_out,test_in):
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(train_in, train_out, 200);
    test_predict, prob_test = adaboost.adaClassify(test_in, classifierArray);  # 测试测试集
    return test_predict