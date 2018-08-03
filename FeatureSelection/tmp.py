import pandas as pd
import  numpy as np
import sklearn.metrics  as skm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

'''
根据阈值进行二分类
'''
def binaryclassify(datamat, thresh):
    label=[]
    num_sample=np.shape(datamat)[0]
    for i in range(num_sample):
        if (datamat[i] < thresh):
            tmp = 0
        else:
            tmp = 1
        label.append(tmp)
    return label

'''
根据ROC曲线确定分类阈值，敏感性和特异性相差最小的位置
'''
def findthresh(datamat, label):
    fpr, tpr, thresholds = skm.roc_curve(label, datamat, pos_label=1)
    n=np.shape(fpr)[0]
    x=np.linspace(0, n-1, n)
    # plt.plot(x,fpr,'b-',x,(1-tpr),'r-')
    # plt.show()
    arg=abs(fpr+tpr-1)
    minindex = np.argmin(arg)
    thresh=thresholds[minindex]
    return thresh

'''
计算分类结果的评价指标
'''
def evaluatemodel(y_true, y_predict,Mews):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel();
    TPR = tp / (tp + fn);
    SPC = tn / (fp + tn);
    PPV = tp / (tp + fp);
    NPV = tn / (tn + fn);
    ACC = (tp + tn) / (tn + fp + fn + tp);

    AUC = roc_auc_score(y_true, Mews)
    BER = 0.5 * ((1 - TPR) + (1 - SPC))

    return [[TPR, SPC, PPV, NPV, ACC, AUC, BER]], [[tn, fp, fn, tp]]

data = pd.read_csv('outlier25000.csv')
labelMat = data['classlabel']
# dataMat = data.iloc[:, 0:60]
dataMat=data.drop('classlabel',axis=1)
# dataMat=data.drop(['vaso','saps','sapsii','sapsii_prob','lods',
#                       'oasis','oasis_prob','mingcs','apsiii_prob','apsiii',
#                       'vent','sofa','hospmor','classlabel','icu_length_of_stay',] ,axis=1)
# dataMat=dataMat.drop(['gender'] ,axis=1)
# 去除假设检验不通过的
delnames = [ 'wbc_min', 'wbc_avg', 'wbc_max', 'ph_avg', 'ph_min','ph_max',
             'platelet_max','lactate_max','lactate_avg','sirs','saps',
             'diasbp_max','sysbp_max','meanbp_max','mingcs','BMI']
# delnames = [ 'po2_avg', 'pco2_max', 'ph_min', 'ph_avg', 'ph_min','rbc_max',
#              'creatinine_min','creatinine_max','creatinine_avg','sirs','saps',
#              'bun_max','bun_avg','pt_min','inr_min','meanbp_mean',
#              'resprate_mean','resprate_min','spo2_max',
#              'spo2_min','si_max','diuretic']#16273个样本假设检验不通过的
dataMat = dataMat.drop(delnames, axis=1)
featurenames=dataMat.keys()

dataMat=dataMat['sofa']
thre=[]
evaluate_test=[]
evaluate_train=[]

"""
十折交叉验证，利用MEWS对AHE进行分类，并计算各折的评价指标，
将验证集十折的评价指标的结果存储到'AHE/Featureselection_MachineLearning/BER /BER_MEWS.csv'
"""
# skf=StratifiedShuffleSplit(n_splits=5)
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(dataMat, labelMat):
    print("%s %s" % (train, test))
    predict_ensemble = []  # 存放15个模型对测试集的预测结果
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    # train_in=train_in.reshape(-1,1)#只有一个特征值，过采样前特殊处理


    trainset = np.c_[train_in, train_out]
    trainset = np.random.permutation(trainset)
    train_pos = trainset[trainset[:, 1] == 1]  # 训练集中的全部阳性样本
    train_neg = trainset[trainset[:, 1] == 0]  # 训练集中的全部阴性样本
    num_neg = np.shape(train_neg)[0]  # 阴性样本个数
    num_pos = np.shape(train_pos)[0]  # 阳性样本个数
    n_split = int(num_neg / num_pos)  # 划分多少个模型

    for i in range(15):  # 阴性样本分成15份，分别与相同的阳性样本构成平衡的训练集，训练15个模型
        print(i)
        if i < 14:
            neg = train_neg[(i * num_pos + 1):((i + 1) * num_pos), :]
        else:
            neg = train_neg[(i * num_pos + 1):num_neg, :]
        train_i = np.r_[neg, train_pos]
        train_i = np.random.permutation(train_i)

        thre_tmp = findthresh(train_i[:, 0], train_i[:, 1])
        thre.append(thre_tmp)

        train_predict = binaryclassify(train_i[:, 0], thre_tmp)
        test_in = np.array(test_in)
        pre_i = binaryclassify(test_in, thre_tmp)

        predict_ensemble.append(pre_i)

    predict_ensemble = np.array(predict_ensemble)
    # print()

    sum_pre = np.sum(predict_ensemble, axis=0)
    tmp = sum_pre.copy()
    sum_pre[tmp > 1] = 1
    sum_pre[tmp < 2] = 0
    test_predict = sum_pre  # 测试集的预测结果

    # sum_prob = []
    # for j in range(np.shape(test_in)[0]):
    #     colj = predict_prob[:, j]
    #     if sum_pre[j] == 1:
    #         tmp_prob = np.mean(colj[colj >= 0.5])
    #     else:
    #         tmp_prob = np.mean(colj[colj < 0.5])
    #     # if tmp_prob == np.nan:
    #     #     print('test')
    #     sum_prob.append(tmp_prob)
    # sum_prob = np.array(sum_prob)
    # proba_test = sum_prob  # 测试集概率预测结果

    # test_predict=clf.predict(test_in)
    # proba_test=clf.predict_proba(test_in)

    # train_predict=clf.predict(train_in)
    # proba_train=clf.predict_proba(train_in)

    # len_train = np.shape(train_in)[0]
    # len_test = np.shape(test_in)[0]
    # test1, test2 = ann.evaluatemodel(train_out, train_predict, proba_train[:,1])  # test model with trainset
    # evaluate_train.extend(test1)
    # prenum_train.extend(test2)


    test3, test4 = evaluatemodel(test_out, test_predict, test_in)  # test model with testset
    evaluate_test.extend(test3)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
Result_test.to_csv('BER_MEWS.csv')
Result_test.boxplot()
plt.show()

mean_test = np.mean(evaluate_test, axis=0)
std_test = np.std(evaluate_test, axis=0)
evaluate_test.append(mean_test)
evaluate_test.append(std_test)

evaluate_test = np.array(evaluate_test)



print('test')