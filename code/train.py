import pandas as pd
import numpy as np
from code.util import SMAPE, MAPE, crossValidation, gridSearch, kFoldCV
from code.model import buildTrainModel
from sklearn.externals import joblib
import matplotlib as mpl
mpl.use('Agg')
import xgboost as xgb
import matplotlib.pyplot as plt

def train(features, trainPath, index, saveName):

    dataDF = pd.read_csv(trainPath, dtype={'link_ID':str})
    print("original dataset columns:", dataDF.columns)
    print("original data num is:", len(dataDF))

    #dataDF = dataDF.dropna(axis=0) # 去除空值记录
    #print("after drop na data num is:", len(dataDF))

    trainX = dataDF[features]
    trainY = dataDF['travel_time']

    trainY = np.log1p(trainY)

    print("trainx shape", trainX.values.shape)
    print("trainY shape", trainY.values.shape)

    rf = buildTrainModel(modelIndex=index)
    #rf = gridSearch(trainX, trainY, modelIndex=modelIndex)
    rf.fit(trainX, trainY)

    #scores, skscores = crossValidation(trainX, trainY, index)
    scores = kFoldCV(trainX, trainY, index, k=5)
    print("cross validation scores:", scores)
    #print("sklearn cross validation scores:", skscores)


    if index == 1 or index == 2:
        print("feature score ", pd.DataFrame(rf.feature_importances_))
    if index == 3:
        #print("feature score ", pd.DataFrame(rf.feature_importances_))
        xgb.plot_importance(rf)
        plt.savefig("../model/importance3.jpg")
    saveSuffix = "../model/"
    joblib.dump(rf, saveSuffix+saveName)
    return rf

def predict(features, testPath, modelPath, resultPath):

    rf = joblib.load(modelPath)

    testDataDF = pd.read_csv(testPath, dtype={'link_ID':str})

    testX = testDataDF[features]

    print("testx shape", testX.values.shape)

    ans = rf.predict(testX)
    ans = np.expm1(ans)
    #np.set_printoptions(threshold=np.nan)
    result = pd.DataFrame()
    result["link_ID"] = testDataDF["link_ID"]
    result["date"] = testDataDF["date"]
    result['time_interval'] = testDataDF['time_interval']
    result["travel_time"] = ans


    # submission = pd.read_csv("../data/submission2.txt", delimiter="#")
    # print(submission)
    # submission = pd.merge(submission, result)

    result.to_csv(resultPath, index=False, sep="#", header=False)

if __name__ == "__main__":

    # 选择特征
    # features = [
    #     #'link_ID',
    #     # 'date', 'time_interval', 'travel_time',
    #     #'in_links', 'out_links','link_class',
    #     'encode_link_ID',
    #
    #     'month',
    #     #'day',
    #    'weekday', 'hour', 'minute',
    #     #'morning_peak', 'evening_peak',
    #
    #     'length', 'width',  'in_links_num',
    #    'in_length_sum', 'in_length_diff', 'in_width_sum', 'in_width_diff',
    #    'out_links_num', 'out_length_sum', 'out_length_diff', 'out_width_sum',
    #    'out_width_diff',
    #     #
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
        # 'last_mean_10',
        # 'last_mean_15', 'last_mean_20',
        # 'last_mean_5',
        # 'max', 'max_10',
        # 'mean', 'median',
        # 'median_10',
        # 'min',
        # 'min_10',
        # 'range', 'range_10', 'std', 'std_10',
        # 'trend_1', 'trend_2', 'trend_3'
        ]

    # 模型序号  # 输入参数为模型序号，1是GBDT，2是随机森林,3是xgboost,
    # 4是adaboost回归，5是多层感知器，6是k近邻回归 7是lightGBM, 8是模型融合stacking
    modelIndex = 3

    testPath = "../data/testB_1.csv"
    trainPath = "../data/trainB_1.csv"
    modelPath = "../model/xgboost1.m"
    resultPath = "../result/result1.txt"


    rf = train(features, trainPath, modelIndex, modelPath)

    predict(features, testPath, modelPath, resultPath)


