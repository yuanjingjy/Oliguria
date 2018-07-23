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
from sklearn.neural_network import MLPClassifier  # import the classifier
from sklearn import  svm
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as  pd  # python data analysis
import matplotlib.pyplot as plt
import  xgboost as xg

data = pd.read_csv('outlier25789.csv')
labelMat = data['classlabel']
# dataMat = data.iloc[:, 0:60]
dataMat=data.drop('classlabel',axis=1)
# dataMat=data.drop(['vaso','saps','sapsii','sapsii_prob','lods',
#                       'oasis','oasis_prob','mingcs','apsiii_prob','apsiii',
#                       'vent','sofa','hospmor','classlabel','icu_length_of_stay',] ,axis=1)
# dataMat=dataMat.drop(['gender'] ,axis=1)
# 去除假设检验不通过的
# delnames = ['pco2_avg', 'ph_avg', 'wbc_min', 'wbc_avg', 'wbc_max',
#             'rbc_max', 'rbc_avg', 'rbc_min', 'ph_avg', 'platelet_avg', 'platelet_max',
#             'creatinine_min', 'creatinine_avg', 'bun_min', 'bun_max',
#             'bun_avg', 'pt_min', 'pt_avg', 'inr_min', 'heartrate_min', 'diasbp_max',
#             'meanbp_mean', 'resprate_mean', 'resprate_min', 'spo2_mean',
#             'spo2_max', 'spo2_min', 'BMI']
# dataMat = dataMat.drop(delnames, axis=1)
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
# dataMat, labelMat = RandomOverSampler().fit_sample(dataMat, labelMat)
# dataMat, labelMat = RandomUnderSampler().fit_sample(dataMat,labelMat)
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
    # clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1,
    #                        fit_intercept=True, intercept_scaling=1, class_weight='balanced',
    #                        random_state=None, solver='lbfgs', max_iter=10000,
    #                        multi_class='ovr',  warm_start=True)
    # clf = MLPClassifier(hidden_layer_sizes=(60,),#9,10
    #                     activation='tanh',
    #                     shuffle=True,
    #                     solver='sgd',
    #                     alpha=1e-6,
    #                     batch_size=5,
    #                     early_stopping=True,
    #                     max_iter=1000
    #                     )
    # clf=xg.XGBClassifier()
    # clf = svm.SVC(C=43, kernel='rbf', gamma='auto',
    #               shrinking=True, probability=True, tol=0.0001,
    #               cache_size=1000, max_iter=-1, class_weight='balanced',
    #               decision_function_shape='ovr', random_state=None
    #               )
    clf=AdaBoostClassifier( n_estimators=200,algorithm='SAMME',  random_state=200)
    # clf=BaggingClassifier(n_estimators=200,max_samples=1.0,max_features=1.0,
    #                       bootstrap=True,bootstrap_features=False,random_state=200)
    trainset = np.c_[train_in, train_out]
    trainset=np.random.permutation(trainset)
    train_pos = trainset[trainset[:,64] == 1]#训练集中的全部阳性样本
    train_neg = trainset[trainset[:, 64] == 0]#训练集中的全部阴性样本
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
        clf.fit(train_i[:,0:64],train_i[:,64])
        pre_i=clf.predict(test_in[:,0:64])
        pre_pro_i=clf.predict_proba(test_in)
        predict_ensemble.append(pre_i)
        predict_prob.append(pre_pro_i[:, 1])
    predict_prob = np.array(predict_prob)
    predict_ensemble = np.array(predict_ensemble)
    # print()

    sum_pre = np.sum(predict_ensemble,axis=0)
    tmp=sum_pre.copy()
    sum_pre[tmp > 9] = 1
    sum_pre[tmp < 10] = 0
    test_predict = sum_pre#测试集的预测结果

    sum_prob = []
    for j in range(np.shape(test_in)[0]):
        colj = predict_prob[:, j]
        if sum_pre[j] == 1:
            tmp_prob = np.mean(colj[colj >= 0.5])
        else:
            tmp_prob = np.mean(colj[colj < 0.5])
        if tmp_prob == nan:
            print('test')
        sum_prob.append(tmp_prob)
    sum_prob = np.array(sum_prob)
    proba_test = sum_prob#测试集概率预测结果


    # test_predict=clf.predict(test_in)
    # proba_test=clf.predict_proba(test_in)

    # train_predict=clf.predict(train_in)
    # proba_train=clf.predict_proba(train_in)


    # len_train = np.shape(train_in)[0]
    # len_test = np.shape(test_in)[0]
    # test1, test2 = ann.evaluatemodel(train_out, train_predict, proba_train[:,1])  # test model with trainset
    # evaluate_train.extend(test1)
    # prenum_train.extend(test2)

    test3, test4 = ann.evaluatemodel(test_out, test_predict, proba_test)  # test model with testset
    evaluate_test.extend(test3)
    prenum_test.extend(test4)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
Result_test.to_csv('BER_LR_ks.csv')
Result_test.boxplot()
plt.show()

# mean_train = np.mean(evaluate_train, axis=0)
# std_train = np.std(evaluate_train, axis=0)
# evaluate_train.append(mean_train)
# evaluate_train.append(std_train)

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)

# evaluate_train = np.array(evaluate_train)
evaluate_test = np.array(evaluate_test)
# prenum_train = np.array(prenum_train)
prenum_test = np.array(prenum_test)

# evaluate_train_mean = np.mean(evaluate_test, axis=0)

print(evaluate_test)