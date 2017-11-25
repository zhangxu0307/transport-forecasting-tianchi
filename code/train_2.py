import pandas as pd
import numpy as np
from code.util import SMAPE, MAPE, crossValidation, gridSearch, kFoldCV
from code.model import buildTrainModel
from sklearn.externals import joblib
import matplotlib as mpl

mpl.use('Agg')
import xgboost as xgb
import matplotlib.pyplot as plt


def train(features, trainPath, testPath, index, saveName, onehotFeature):
    trainDF = pd.read_csv(trainPath, dtype={'link_ID': str})
    print("original dataset columns:", trainDF.columns)
    testDF = pd.read_csv(testPath, dtype={'link_ID': str})

    print(len(trainDF))

    totalDF = pd.concat([trainDF, testDF], axis=0)
    print(len(totalDF))

    trainX = totalDF[features]
    for catgoryFeature in onehotFeature:
        Onehot = pd.get_dummies(trainX[catgoryFeature], prefix=catgoryFeature, sparse=True)
        trainX = pd.concat([trainX, Onehot], axis=1)
    trainX = trainX[trainX["month"] < 6]
    trainX = trainX.drop(onehotFeature, axis=1)
    print(trainX.columns)

    # label 做对数变换
    trainY = trainDF['travel_time']
    trainY = np.log1p(trainY)

    print("trainx shape", trainX.values.shape)
    print("trainY shape", trainY.values.shape)

    rf = buildTrainModel(modelIndex=index)
    # rf = gridSearch(trainX, trainY, modelIndex=modelIndex)
    rf.fit(trainX, trainY)

    # scores, skscores = crossValidation(trainX, trainY, index)
    scores = kFoldCV(trainX, trainY, modelIndex, k=5)
    print("cross validation scores:", scores)
    # print("sklearn cross validation scores:", skscores)


    if index == 1 or index == 2:
        print("feature score ", pd.DataFrame(rf.feature_importances_))
    if index == 3:
        # print("feature score ", pd.DataFrame(rf.feature_importances_))
        xgb.plot_importance(rf)
        plt.savefig("../model/importance3.jpg")
    saveSuffix = "../model/"
    joblib.dump(rf, saveSuffix + saveName)
    return rf


def predict(features, testPath, modelPath, resultPath, onehotFeature):
    rf = joblib.load(modelPath)

    trainDF = pd.read_csv(trainPath, dtype={'link_ID': str})
    testDF = pd.read_csv(testPath, dtype={'link_ID': str})

    totalDF = pd.concat([trainDF, testDF], axis=0)

    testX = totalDF[features]
    for catgoryFeature in onehotFeature:
        Onehot = pd.get_dummies(testX[catgoryFeature], prefix=catgoryFeature, sparse=True)
        testX = pd.concat([testX, Onehot], axis=1)
    testX = testX[testX["month"] >= 6]
    testX = testX.drop(onehotFeature, axis=1)
    print(testX.columns)

    ans = rf.predict(testX)
    ans = np.expm1(ans)
    # np.set_printoptions(threshold=np.nan)
    result = pd.DataFrame()
    result["link_ID"] = testDF["link_ID"]
    result["date"] = testDF["date"]
    result['time_interval'] = testDF['time_interval']
    result["travel_time"] = ans

    # submission = pd.read_csv("../data/submission2.txt", delimiter="#")
    # print(submission)
    # submission = pd.merge(submission, result)

    result.to_csv(resultPath, index=False, sep="#", header=False)


if __name__ == "__main__":
    # 选择特征
    # features = [
    #     # 'link_ID',
    #     # 'date', 'time_interval', 'travel_time',
    #     # 'in_links', 'out_links','link_class',
    #     'encode_link_ID',
    #
    #     'month',
    #     # 'day',
    #     'weekday', 'hour', 'minute',
    #     # 'morning_peak', 'evening_peak',
    #
    #     'length', 'width', 'in_links_num',
    #     'in_length_sum', 'in_length_diff', 'in_width_sum', 'in_width_diff',
    #     'out_links_num', 'out_length_sum', 'out_length_diff', 'out_width_sum',
    #     'out_width_diff',
    #     # #
    #     'mean', 'last_mean_10', 'last_mean_20', 'last_mean_30', 'median', 'min', 'max',
    #     'std', 'range',
    # ]

    features = [
        # 'link_ID', 'date', 'time_interval', 'travel_time', 'link_class','in_links', 'out_links','satrt_date_time',
        'month',
        # 'day','start_date_time',
        'weekday',
        'is_weekend',
        #'holiday',
        'hour', 'minute',
        'morning_peak', 'evening_peak',
        'length', 'width',
        'in_links_num',
        'in_length_sum',
        'in_length_diff', 'in_width_sum', 'in_width_diff',
         'out_links_num',
        'out_length_sum', 'out_length_diff', 'out_width_sum', 'out_width_diff',
        'encode_link_ID',
        'last_mean_10',
        'last_mean_15', 'last_mean_20',
        'last_mean_5',
        'max', 'max_10',
        'mean', 'median',
         'median_10',
        'min',
        'min_10',
        'range', 'range_10', 'std', 'std_10',
        'trend_1', 'trend_2', 'trend_3']

    onehotFeature = [
        'month',
        # 'day',
         'encode_link_ID',
        #'is_weekend','holiday',
        'weekday', 'hour', 'minute',]

    # 模型序号  # 输入参数为模型序号，1是GBDT，2是随机森林,3是xgboost,
    # 4是adaboost回归，5是多层感知器，6是k近邻回归 7是lightGBM, 8是模型融合stacking
    modelIndex = 3

    testPath = "../data/testB_2.csv"
    trainPath = "../data/trainB_2.csv"
    modelPath = "../model/xgboost2.m"
    resultPath = "../result/result2.txt"

    rf = train(features, trainPath, testPath, modelIndex, modelPath, onehotFeature)

    predict(features, testPath, modelPath, resultPath, onehotFeature)


