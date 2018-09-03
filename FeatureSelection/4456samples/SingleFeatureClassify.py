"""
Description:
    利用单一特征进行分类：sofa、apsiii、creatine_avg，处理流程：
        1.处理不平衡数据：每一折中，阴性样本数目是阳性样本数目的15倍，因此将阴性样本分为15份，每份分别与阳性样本进行组合。
            进行模型训练，寻找分类阈值。
        2.对于每一组平衡处理后的训练数据，通过绘制roc曲线，得到敏感性、特异性平衡处的分类阈值，对该折中不平衡的测试集数据进行预测
        3.每一折中用15个模型各对测试集进行一次预测，并通过对15个模型预测结果进行投票，得到最终的预测结果，如果有7个以上的模型判定为0类，
            则认为该样本属于0类。
        4.根据用单一特征值进行分类得到的结果，计算模型预测结果的敏感性、特异性、PPV、NPV等参数，因为计算AUC的过程中需要用到
            预测的概率值，此处直接将单一的特征值和实际标签代入AUC计算函数
Inputs:
    'outlier25789.csv'：24小时标准定义少尿的特征值矩阵
    选择的单一特征包括：sofa评分、apsiii评分、肌酐值
Outputs:
    [[TPR, SPC, PPV, NPV, ACC, AUC, BER]]
    [[tn, fp, fn, tp]]

"""
import pandas as pd
import  numpy as np
import sklearn.metrics  as skm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.over_sampling import  RandomOverSampler

'''
Description:
    根据阈值进行二分类，输入数据大于阈值，则认为该样本属于1类，否则，该样本属于0类
Inputs:
    datamat：用来进行分类的特征
    thresh：分类的阈值
Output:
    label：分类后得到的标签
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
Description:
    针对用单一特征值进行分类的情况，根据ROC曲线确定分类阈值：敏感性和特异性相差最小的位置
Inouts：
    datamat：用于分类的单一特征值，可以是疾病严重程度评分等
    label：实际标签
'''
def findthresh(datamat, label):
    fpr, tpr, thresholds = skm.roc_curve(label, datamat, pos_label=1)#画ROC曲线，得到分类的敏感性、特异性以及对应的阈值
    n=np.shape(fpr)[0]#划分的节点数目
    x=np.linspace(0, n-1, n)
    # plt.plot(x,fpr,'b-',x,(1-tpr),'r-')
    # plt.show()
    arg=abs(fpr+tpr-1)
    minindex = np.argmin(arg)
    thresh=thresholds[minindex]
    return thresh

'''
Description:
    针对单一特征进行分类问题，计算分类结果的评价指标
Inputs：
    y_true：实际标签
    y_predict：用单一的特征进行分类的分类结果
    Mews：单一特征的原始特征，用来当作概率值计算AUC
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

data = pd.read_csv('./data/outlier16273.csv')
labelMat = data['classlabel']
dataMat=data['creatinine_avg']
thre=[]
evaluate_test=[]
evaluate_train=[]

"""
十折交叉验证，利用MEWS对AHE进行分类，并计算各折的评价指标，
将验证集十折的评价指标的结果存储到'AHE/Featureselection_MachineLearning/BER /BER_MEWS.csv'
"""
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(dataMat, labelMat):
    print("%s %s" % (train, test))
    predict_ensemble = []  # 存放15个模型对测试集的预测结果
    train_in = dataMat[train]
    test_in = dataMat[test]
    train_out = labelMat[train]
    test_out = labelMat[test]
    train_in=train_in.reshape(-1,1)#只有一个特征值，过采样前特殊处理
    train_in,train_out = RandomOverSampler().fit_sample(train_in,train_out)

    test_in = np.array(test_in)
    test_out= np.array(test_out)

    thre_tmp = findthresh(train_in, train_out)
    thre.append(thre_tmp)

    train_predict = binaryclassify(train_in, thre_tmp)
    test_predict = binaryclassify(test_in, thre_tmp)

    test1, test2 = evaluatemodel(train_out, train_predict, train_in)  # test model with trainset
    evaluate_train.extend(test1)

    test3, test4 = evaluatemodel(test_out, test_predict, test_in)  # test model with testset
    evaluate_test.extend(test3)

Result_test = pd.DataFrame(evaluate_test, columns=['TPR', 'SPC', 'PPV', 'NPV', 'ACC', 'AUC', 'BER'])
Result_test.to_csv('BER_MEWS.csv')
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


evaluate_train_mean = np.mean(evaluate_test, axis=0)


print('test')