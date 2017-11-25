import pandas as pd

# 平均规则
def calcGroupMean():

    trainPath = "../data/gy_contest_link_traveltime_training_data.txt"
    trainFeaturePath = "../data/train_feature.csv"
    train = pd.read_csv(trainPath, delimiter=";")
    trainFeature = pd.read_csv(trainFeaturePath)

    trainFeatureReginal = trainFeature.loc[trainFeature['hour'] == 8] # 选取8点钟的历史数据分组平均
    print(len((trainFeatureReginal)))


    group = trainFeatureReginal.groupby("link_ID")
    mean = group.mean()
    mean.to_csv("../data/mean_2.csv", columns=["travel_time"])
    # mean.csv是全局做平均，mean_2.csv是8-9点历史数据做平均

# 生成平均结果提交文件
def meanSubmit():

    mean = pd.read_csv("../data/mean_2.csv")
    submission = pd.read_csv("../data/submission2.txt", delimiter="#")

    print(mean.columns)
    print(submission.columns)
    submission = submission.drop("travel_time1", axis=1) # 去掉结果文件中占位的travel_time

    result = pd.merge(submission, mean, on="link_ID")
    result.to_csv("../result/mean_res2.txt", sep="#", index=False, header = False) # 去掉头部


if __name__ == "__main__":
    calcGroupMean()
    meanSubmit()