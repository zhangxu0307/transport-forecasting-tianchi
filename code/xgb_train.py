import xgboost as xgb
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# 计算SMAPE，主要是应对ytrue是0的情况
def SMAPE(preds, dtrain):
    labels = dtrain.get_label()
    return ("SMAPE", np.mean(np.abs((preds-labels) / (preds+labels))) * 2.0)

# 计算MAPE，需要提前丢弃ytrue是0的情况
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 交叉验证
def crossValidation(trainX, trainY, cvRate = 0.96, cvEpoch = 20):

    scores = []
    for i in range(cvEpoch):
        X, Y = shuffle(trainX, trainY) # 打乱数据
        offset = int(X.shape[0] * cvRate)
        X_train, y_train = X[:offset], Y[:offset]
        X_test, y_test = X[offset:], Y[offset:]

        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
        dtest = xgb.DMatrix(X_test, missing=np.nan)
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.005
        params["min_child_weight"] = 6
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.9
        params["scale_pos_weight"] = 1
        params["silent"] = 1
        params["max_depth"] = 10

        plst = list(params.items())

        num_rounds = 800

        rf = xgb.train(plst, dtrain, num_rounds)

        pred = rf.predict(dtest)
        pred = np.rint(pred)
        acc = SMAPE(y_test, pred)
        scores.append(acc)
    print("score mean:", np.mean(scores))
    print("score std:", np.std(scores))
    return scores

def xgbTrain(features):

    trainSet = pd.read_csv("../data/train2.csv", dtype={'link_ID':str})
    testSet = pd.read_csv("../data/test2.csv", dtype={'link_ID':str})

    train = trainSet[features]
    test = testSet[features]

    trainNum = train.shape[0]
    testNum = test.shape[0]

    Y = trainSet["travel_time"].values # 提取label
    Y = np.log1p(Y) # 取对数

    #train = train.drop("travel_time", axis=1)

    trainData = train.values
    testData = test.values

    valRate = 0.8
    trainSampleNum = int(valRate * trainNum)
    df_columns = train.columns

    trainX = trainData[:trainSampleNum, :]
    trainY = Y[:trainSampleNum]

    valX = trainData[trainSampleNum:, :]
    valY = Y[trainSampleNum:]

    testX = testData

    print("trainx shape:", trainX.shape)
    print("trainy shape:", trainY.shape)
    print("valx shape:", valX.shape)
    print("valy shape:", valY.shape)
    print("testx shape", testX.shape)

    dtrain_all = xgb.DMatrix(trainData, Y, feature_names=df_columns)
    dtrain = xgb.DMatrix(trainX, trainY, feature_names=df_columns)
    dval = xgb.DMatrix(valX, valY, feature_names=df_columns)
    dtest = xgb.DMatrix(testX, feature_names=df_columns)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        #'eval_metric': 'SMAPE',
        'silent': 1
    }

    # Uncomment to tune XGB `num_boost_rounds`
    partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                              early_stopping_rounds=20, verbose_eval=20, feval=SMAPE)

    num_boost_round = partial_model.best_iteration

    # fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    # xgb.plot_importance(partial_model, max_num_features=50, height=0.5)
    # plt.show()

    model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)

    ylog_pred = model.predict(dtest)
    y_pred = np.expm1(ylog_pred)

    testDataDF = pd.read_csv("../data/test2.csv", dtype={'link_ID': str})

    result = pd.DataFrame()
    result["link_ID"] = testDataDF["link_ID"]
    result["date"] = testDataDF["date"]
    result['time_interval'] = testDataDF['time_interval']
    result["travel_time"] = y_pred
    print(result.head(10))

    result.to_csv("../result/result8.txt", index=False, sep="#", header=False)


if __name__ == "__main__":
    features = [ #'link_ID',
        # 'date', 'time_interval',
        #'in_links', 'out_links','link_class',
        #'travel_time',
        'encode_link_ID',
        'month', 'day',
       'weekday', 'hour', 'minute',
        'morning_peak', 'evening_peak',
        'length', 'width',  'in_links_num',
       'in_length_sum', 'in_length_diff', 'in_width_sum', 'in_width_diff',
       'out_links_num', 'out_length_sum', 'out_length_diff', 'out_width_sum',
       'out_width_diff',
        #
        'mean', 'last_mean_10', 'last_mean_20', 'last_mean_30', 'median', 'min', 'max',
        'std', 'range',
                ]
    xgbTrain(features)




