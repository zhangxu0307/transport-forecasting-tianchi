import pandas as pd
import matplotlib.pyplot as plt

# linkInfoPath = "../data/gy_contest_link_info.txt"
# topPath = "../data/gy_contest_link_top1.txt"
# trainPath = "../data/gy_contest_link_traveltime_training_data.txt"
# linkInfo = pd.read_csv(linkInfoPath, delimiter=";")
#
# print(linkInfo.head(100))
# link = linkInfo["link_ID"].drop_duplicates()
# print(len(link))
#
# linkClass = linkInfo["link_class"].drop_duplicates()
# print(len(linkClass))

# topInfo = pd.read_csv(topPath)


# train = pd.read_csv(trainPath, delimiter=";")
# print(train.head(100))
# print(len(train))
# date = train["time_interval"].drop_duplicates().sort_values()
# print(date)
trainPath = "../data/gy_contest_traveltime_training_data_second.txt"
timePath = "../data/train_8-9_clock.csv"
train = pd.read_csv("../data/test_history_feature_B.csv", dtype={'link_ID':str}, delimiter=",")
#
# train = pd.read_csv(trainPath, dtype={'link_ID':str}, delimiter=",")

#train = train[train["hour"] == 8]
#train = pd.read_csv("../result/result8.txt", dtype={'link_ID':str}, delimiter="#")
#train.columns = ["link_ID", "date","time_interval", "travel_time"]
#trainsix = train[train["month"] >= 6]
print(train.columns)
print(len(train["link_ID"].drop_duplicates()))
print (len(train))
#train = train[train["month"] >= 6]
#train = train[train["hour"] <= 12]
print(len(train.dropna(axis=0)))
print(len(train))
#train.to_csv("../data/train7.csv", index=False)

# train[train["travel_time"] < 250].hist(bins=100, column="travel_time")
# plt.savefig("../data/divide_res1_hist.jpg")

# train = train.dropna()
# print(len(train))


