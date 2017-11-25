import xgboost as xgb
import numpy as np
import pandas as pd
from code_2.util import kFoldCV

def SMAPE(preds, dtrain):
    labels = dtrain.get_label()
    return ("SMAPE", np.mean(np.abs((preds-labels) / (preds+labels))) * 2.0)

def buildXGBModel(train, Y, test):


    train = train.values
    testData = test.values
    trainNum = len(train)

    valRate = 0.8
    trainSampleNum = int(valRate * trainNum)
    #df_columns = train.columns

    trainX = train[:trainSampleNum, :]
    trainY = Y[:trainSampleNum]

    valX = train[trainSampleNum:, :]
    valY = Y[trainSampleNum:]

    testX = testData

    print("trainx shape:", trainX.shape)
    print("trainy shape:", trainY.shape)
    print("valx shape:", valX.shape)
    print("valy shape:", valY.shape)
    #print("testx shape", testX.shape)

    dtrain_all = xgb.DMatrix(trainX, trainY)
    dtrain = xgb.DMatrix(trainX, trainY)
    dval = xgb.DMatrix(valX, valY)
    dtest = xgb.DMatrix(testX)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'objective': 'reg:linear',
        #'eval_metric': 'rmse',
        'silent': 1
    }

    # Uncomment to tune XGB `num_boost_rounds`
    partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                              early_stopping_rounds=20, verbose_eval=20, feval=SMAPE)

    num_boost_round = partial_model.best_iteration
    print("best boost round:", num_boost_round)

    scores = kFoldCV(train, Y, modelIndex=3,modelParameter=num_boost_round, k=5, logFlag=True)
    print("cross validation scores:", scores)

    model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

    ylog_pred = model.predict(dtest)
    y_pred = np.expm1(ylog_pred)

    return y_pred

    # testDataDF = pd.read_csv("../data/test2.csv", dtype={'link_ID': str})
    #
    # result = pd.DataFrame()
    # result["link_ID"] = testDataDF["link_ID"]
    # result["date"] = testDataDF["date"]
    # result['time_interval'] = testDataDF['time_interval']
    # result["travel_time"] = y_pred
    # #print(result.head(10))
    #
    # result.to_csv("../result/divide_xgb_result1.txt", index=False, sep="#", header=False)
