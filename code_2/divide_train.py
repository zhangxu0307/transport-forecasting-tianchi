import pandas as pd
import numpy as np
from code_2.model import buildTrainModel
from code_2.util import kFoldCV
import xgboost as xgb
import matplotlib.pyplot as plt
from code_2.xgb_model import buildXGBModel


features = [
        # 'link_ID', 'date', 'time_interval', 'travel_time', 'link_class','in_links', 'out_links','satrt_date_time',
        'month',
        #'day','start_date_time',
        'weekday',
        'is_weekend', 'holiday',
        'hour', 'minute',
        'morning_peak', 'evening_peak',
        'length', 'width',
        # 'in_links_num',
        # 'in_length_sum',
        # 'in_length_diff', 'in_width_sum', 'in_width_diff',
        # 'out_links_num',
        # 'out_length_sum', 'out_length_diff', 'out_width_sum', 'out_width_diff',
        #'encode_link_ID',
        'last_mean_10',
        'last_mean_15', 'last_mean_20',
         'last_mean_5',
         'max', 'max_10',
         'mean', 'median',
        'median_10',
        'min',
        'min_10',
         'range', 'range_10', 'std', 'std_10',
        'trend_1', 'trend_2', 'trend_3'
        ]


linkInfoPath = "../data/gy_contest_link_info.txt"
linkInfo = pd.read_csv(linkInfoPath, delimiter=";")
linkIDs = linkInfo["link_ID"]

trainPath = "../data/train6.csv"
train = pd.read_csv(trainPath, dtype={'link_ID': str})
testPath = "../data/test6.csv"
test = pd.read_csv(testPath, dtype={'link_ID': str})
result = pd.DataFrame()

for id, linkID in enumerate(linkIDs):

    subTrain = train[train["link_ID"] == str(linkID)]

    trainX = subTrain[features]
    trainY = subTrain['travel_time']

    subTest = test[test["link_ID"] == str(linkID)]
    subTestx = subTest[features]

    trainY = np.log1p(trainY)
    print("subtrainx shape", trainX.values.shape)
    print("subtrainY shape", trainY.values.shape)

    # modelIndex = 3
    # rf = buildTrainModel(modelIndex=modelIndex)
    # rf.fit(trainX, trainY)
    #
    # # scores, skscores = crossValidation(trainX, trainY, index)
    # scores = kFoldCV(trainX, trainY, modelIndex, k=5)
    # print("cross validation scores:", scores)
    # # print("sklearn cross validation scores:", skscores)
    #

    # ans = rf.predict(subTestx)
    # ans = np.expm1(ans)


    ans = buildXGBModel(trainX, trainY, subTestx)

    subresult = pd.DataFrame()
    subresult["link_ID"] = subTest["link_ID"]
    subresult["date"] = subTest["date"]
    subresult['time_interval'] = subTest['time_interval']
    subresult["travel_time"] = ans

    result = result.append(subresult, ignore_index=True)

    print("%d link is finished!" %id)

    # submission = pd.read_csv("../data/submission2.txt", delimiter="#")
    # print(submission)
    # submission = pd.merge(submission, result)
resultPath = "../result/divide_result1.txt"
result.to_csv(resultPath, index=False, sep="#", header=False)


