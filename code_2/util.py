import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from code_2.model import buildTrainModel
from sklearn.model_selection import GridSearchCV
import copy
import pandas as pd
from datetime import datetime, timedelta, time

# 计算SMAPE，主要是应对ytrue是0的情况
def SMAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 2.0

# 计算MAPE，需要提前丢弃ytrue是0的情况
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 交叉验证, 默认取96%的样本，20轮
def crossValidation(trainX, trainY, modelIndex, cvRate = 0.96, cvEpoch = 20):

    scores = []
    for i in range(cvEpoch):
        X, Y = shuffle(trainX, trainY) # 打乱数据
        offset = int(X.shape[0] * cvRate)
        X_train, y_train = X[:offset], Y[:offset]
        X_test, y_test = X[offset:], Y[offset:]
        #y_train = np.log(y_train+1)

        rf = buildTrainModel(modelIndex=modelIndex)

        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)

        # 四舍五入成整数
        #pred = np.exp(pred)
        pred = np.rint(pred)
        acc = SMAPE(y_test, pred)
        scores.append(acc)

    # 去除评分中的inf
    scores = [x for x in scores if str(x) != 'nan' and str(x) != 'inf']
    print("score mean:", np.mean(scores))
    print("score std:", np.std(scores))
    skscores = cross_val_score(rf, trainX, trainY, cv=10, scoring="neg_mean_absolute_error")
    print("sklearn cv mean:", skscores.mean())
    print("sklearn cv std:", skscores.std())
    return scores, skscores

# k折交叉验证
def splitData(dataArr, label, k): # 划分k折数据

    if not isinstance(dataArr, np.ndarray): # 若不是array，则是pandas，做处理
        dataArr = dataArr.values
        label = label.values

    dataArr, label = shuffle(dataArr, label, random_state=0)

    datas = []
    labels = []

    m,n = dataArr.shape
    last = m%k  # 计算出最后的残余数，合并到最后的一折里
    other = m-last
    for i in range(0, other-m//k, m//k):
        datas.append(dataArr[i:(i+m//k),:])
        labels.append(label[i:i+m//k])
    datas.append(dataArr[other-m//k:,:])
    labels.append(label[other-m//k:])
    return datas, labels

def kFoldCV(trainX, trainY, modelIndex, modelParameter, k = 10, logFlag=True):

    datas, labels = splitData(trainX, trainY, k)

    res = []
    for i in range(k):
        copydata = copy.deepcopy(datas)  # 备份数据集

        # 生成训练和测试样本
        testArr = copydata[i]
        del copydata[i]
        trainArr = np.vstack(tuple(copydata))

        # 　生成测试和训练标签
        copylabel = copy.deepcopy(labels)
        testLabel = copylabel[i]
        del copylabel[i]
        trainLabel = np.hstack(tuple(copylabel))

        # 测试
        rf = buildTrainModel(modelIndex=modelIndex, modelParameter=modelParameter)
        rf.fit(trainArr, trainLabel)
        pred = rf.predict(testArr)
        #pred = np.rint(pred)

        if logFlag: # 对数变换在做交叉验证时先还原数据
            pred = np.expm1(pred)
            testLabel = np.expm1(testLabel)

        res.append(SMAPE(testLabel, pred))
    print("score mean:", np.mean(res))
    print("score std:", np.std(res))
    return res

# 格子搜索参数
def gridSearch(trainx, trainy, modelIndex):

    parameters = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [150, 200, 250],
                  'subsample': [0.5, 0.7, 0.9, 1.0], 'max_depth': [6, 8, 10], 'max_features': ['sqrt', None]}

    rf = buildTrainModel(modelIndex=modelIndex)
    grid_search = GridSearchCV(rf, parameters, verbose=2, cv=10)

    grid_search.fit(trainx, trainy)

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def createSubmission(): # 生成提交文件格式txt

    linkInfoPath = "../data/gy_contest_link_info.txt"
    linkInfo = pd.read_csv(linkInfoPath, delimiter=";")
    for i in range(132):
        linkid = linkInfo.ix[i, "link_ID"]
        startDate = datetime(year=2016, month=6, day=1, hour=8, minute=0, second=0)
        for i in range(30): # 6月30天
            print(startDate)
            dateDiff = timedelta(days=1)
            startTime = startDate
            for i in range(30): # 1小时内30个时间片

                timeDiff = timedelta(minutes=2)
                timeSeg1 = startTime
                timeSeg2 = startTime+timeDiff

                startTime = timeSeg2

                f = open('../data/submission.txt', 'a') # 此处使用追加模式
                f.writelines(str(linkid)+"#"+datetime.strftime(startDate, "%Y-%m-%d")+"#"+"["+
                             datetime.strftime(timeSeg1, "%Y-%m-%d %H:%M:%S")+","+
                             datetime.strftime(timeSeg2, "%Y-%m-%d %H:%M:%S")+")"+"#"+"0"+"\n")

            startDate = startDate+dateDiff


if __name__ == "__main__":

    createSubmission()
