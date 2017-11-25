import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os

def historyFeature():

    trainPath = "../data/gy_contest_traveltime_training_data_second.txt"
    train = pd.read_csv(trainPath, delimiter=";", dtype={'link_ID':str})
    print("load data finished!")

    #train.columns = ["link_ID", "date", "time_interval", "travel_time"]

    # 解析时间
    train["start_date_time"] = train["time_interval"].apply(lambda x:
                                                      datetime.strptime(x.lstrip("[").split(",")[0],
                                                                        "%Y-%m-%d %H:%M:%S"))
    print("time praser finished!")

    group = train.groupby("link_ID")
    for key, item in group: # 遍历所有的link—ID
        volumeFeature = pd.DataFrame()
        print("link ID", key)
        item = item.sort_values(by="start_date_time")  # 按时间排序
        item.index = pd.to_datetime(item["start_date_time"])  # 设置时间索引，方便resample
        item = item.resample('1H')["travel_time"]  # 一小时采样

        for c, subItem in item: # 在该道路下所有的时间片
            #print(c)
            # print(subItem)

            # 平均值信息
            mean = subItem.mean()
            last_mean_5 = subItem.iloc[-5:].mean()
            last_mean_10 = subItem.iloc[-10:].mean()
            last_mean_20 = subItem.iloc[-20:].mean()
            last_mean_15 = subItem.iloc[-15:].mean()

            # 差分趋势信息
            trend_1 = last_mean_10 - last_mean_5
            trend_2 = last_mean_15 - last_mean_10
            trend_3 = last_mean_20 - last_mean_15

            # 邻近1小时统计信息
            median = subItem.median()
            minNum = subItem.min()
            maxNum = subItem.max()
            rangeNum = maxNum - minNum
            std = subItem.std()

            # 邻近20分钟内统计信息
            median_10 = subItem.iloc[-10:].median()
            minNum_10 = subItem.iloc[-10:].min()
            maxNum_10 = subItem.iloc[-10:].max()
            rangeNum_10 = maxNum_10 - minNum_10
            std_10 = subItem.iloc[-10:].std()

            #print([mean, last_mean_10, last_mean_20, last_mean_30, median, minNum, maxNum, rangeNum, std])

            volumeRecord = pd.Series()
            volumeRecord["link_ID"] = str(key)
            volumeRecord["date"] = c.date()+timedelta(hours=1)
            #volumeRecord["time"] = c.time()
            volumeRecord["satrt_date_time"] = str(c+timedelta(hours=1))
            volumeRecord["hour"] = str((c + timedelta(hours=1)).hour)
            # 小时加1，表示以下的统计特征都是给后面1小时内所有的时间片准备的

            # 平均值特征
            volumeRecord["mean"] = mean
            volumeRecord["last_mean_10"] = last_mean_10
            volumeRecord["last_mean_20"] = last_mean_20
            volumeRecord["last_mean_5"] = last_mean_5
            volumeRecord["last_mean_15"] = last_mean_15

            # 差分趋势特征
            volumeRecord["trend_1"] = trend_1
            volumeRecord["trend_2"] = trend_2
            volumeRecord["trend_3"] = trend_3

            # 邻近1小时内其它统计特征
            volumeRecord["median"] = median
            volumeRecord["min"] = minNum
            volumeRecord["max"] = maxNum
            volumeRecord["range"] = rangeNum
            volumeRecord["std"] = std

            # 邻近20分钟内其它统计特征
            volumeRecord["median_10"] = median_10
            volumeRecord["min_10"] = minNum_10
            volumeRecord["max_10"] = maxNum_10
            volumeRecord["range_10"] = rangeNum_10
            volumeRecord["std_10"] = std_10

            volumeFeature = volumeFeature.append(volumeRecord, ignore_index=True)

        volumeFeature.to_csv("../data/history_files_B/"+str(key)+".csv", index=False)
        print("../data/history_files_B/"+str(key)+".csv"+"has benn finished!\n")

def combineHistoryFeature():

    rootdir = "../data/history_files_B/"
    history = pd.DataFrame()
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            file = pd.read_csv(rootdir+filename, dtype={'link_ID':str})
            history = history.append(file, ignore_index=True)
    history.to_csv("../data/history_feature_B.csv", index=False)



def splitHistoryFeature():

    history = pd.read_csv("../data/history_feature_B.csv", dtype={'link_ID':str})

    history["date"] = history["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

    test = history[history["date"] >= datetime(year=2017, month=6, day=1)] # 取出6月1日以后的作为预测集

    train = history[history["date"] < datetime(year=2017, month=6, day=1)] # 取出6月1日以前的作为训练集

    train.to_csv("../data/train_history_feature_B.csv", index=False)
    test.to_csv("../data/test_history_feature_B.csv", index=False)


if __name__ == "__main__":

    historyFeature()
    combineHistoryFeature()
    splitHistoryFeature()
