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
import scipy


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


"""
Description：
    根据Jarque-Bera值，判断数据是否服从正态分布，当序列服从正态分布时，
    JB统计量也渐进服从正态分布。
Inputs:
    datain:待检验的数据
Outputs:
    JB:Jarque-Bera值
    Pvalue：JB值的假设检验结果
"""
def normtest(datain):
    n=np.shape(datain)[0]
    y=datain-datain.mean()
    M2=np.mean(y**2)
    skew=np.mean(y**3)/(M2**1.5)
    kur=np.mean(y**4)/(M2**2)
    JB=n*((skew**2)/6+((kur-3)**2)/24)
    pvalue=1-scipy.stats.chi2.cdf(JB,df=2)
    return JB,pvalue,skew

"""
Description:
    利用改进zscore方法及Turkey方法识别异常值，两种方法都识别为异常值的认为是异常值
    改进zscore法：根据中位数及距离中位数的偏差来识别异常值
    Turkeys方法：定义IQR=上四分位数-下四分位数
                        上四分位数+3*IQR与下四分位数-3*IQR 范围外的定义为异常值
Input:
    datain:待判断数据
Output:
    dataout:将异常值位置置空后输出的矩阵
"""
def zscore_re(datain):
    #利用改进zscore方法识别异常数据
    diff=datain-np.median(datain,axis=0)
    MAD=np.median(abs(diff),axis=0)
    zscore=(0.6745*diff)/(MAD+0.0001)
    zscore=abs(zscore)
    dataout=datain.copy()
    mask_zscore=zscore>3.5

    #利用Turkey方法识别异常值
    Q1,mid,Q3=np.percentile(datain,(25,50,75),axis=0)
    IQR=Q3-Q1
    out_up=Q3+1.5*IQR
    out_down=Q1-1.5*IQR
    mask_precup=np.maximum(datain,out_up)==datain#超过上限的异常值
    mask_precdown = np.maximum(datain, out_down) == out_down#超过下限的异常值
    mask_prec= np.logical_or(mask_precdown, mask_precup)#逻辑或

    maskinfo=np.logical_and(mask_prec,mask_zscore)#两种方法都识别为异常值的认为是异常值

    dataout[maskinfo]=np.nan#异常值位置置空
    return dataout


def evaluatemodel(y_true, y_predict):
    from sklearn.metrics import confusion_matrix
    #    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    #    from sklearn.metrics import precision_score
    #    from sklearn.metrics import recall_score
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel();
    TPR = tp / (tp + fn);
    SPC = tn / (fp + tn);
    PPV = tp / (tp + fp);
    NPV = tn / (tn + fn);
    ACC = (tp + tn) / (tn + fp + fn + tp);

    #    Accuracy=accuracy_score(y_true,y_predict)
    # AUC = roc_auc_score(y_true, proba)
    #    Precision=precision_score(y_true,y_predict)
    #    Recall=recall_score(y_true,y_predict)
    BER = 0.5 * ((1 - TPR) + (1 - SPC))
    return [[TPR, SPC, PPV, NPV, ACC,  BER]], [[tn, fp, fn, tp]]