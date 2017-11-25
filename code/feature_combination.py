import pandas as pd
from sklearn import preprocessing


def combine_Time_link_feature(linkFeatureFile, trainFeatureFile, testFeatureFile, outputTrainFile, outputTestFile):

    link_feature = pd.read_csv(linkFeatureFile, dtype={'link_ID':str})
    train_time_feature = pd.read_csv(trainFeatureFile, dtype={'link_ID':str})
    test_time_feature = pd.read_csv(testFeatureFile, dtype={'link_ID':str})

    #train_time_feature.rename(columns={"linkID": "link_ID"}, inplace=True)
    #test_time_feature.rename(columns={"linkID": "link_ID"}, inplace=True)

    # print(train_time_feature["link_ID"].drop_duplicates())
    # print("---------------------------------------")
    # print(test_time_feature["link_ID"].drop_duplicates())
    # print(link_feature["link_ID"].drop_duplicates())

    trainSet = pd.merge(train_time_feature, link_feature, on="link_ID")
    testSet = pd.merge(test_time_feature, link_feature, on="link_ID")

    print(trainSet["link_ID"].drop_duplicates())
    print("---------------------------------------")
    print(testSet["link_ID"].drop_duplicates())


    # link ID 编码
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(trainSet["link_ID"])
    trainSet["encode_link_ID"] = labelEncoder.transform(trainSet["link_ID"])
    testSet["encode_link_ID"] = labelEncoder.transform(testSet["link_ID"])

    trainSet.to_csv(outputTrainFile, index=False)
    testSet.to_csv(outputTestFile, index=False)

def mix_history(inputTrainFile, inputTestFile, outputTrainFile, outputTestFile):

    train1 = pd.read_csv(inputTrainFile, dtype={'link_ID':str})
    test1 =  pd.read_csv(inputTestFile, dtype={'link_ID':str})
    print(len(train1))
    print(len(test1))
    print(train1["link_ID"].dtypes)
    print(test1["link_ID"].dtypes)
    print("train1 and test1 load finished!")

    train_history = pd.read_csv("../data/train_history_feature_B.csv", dtype={'link_ID':str})
    test_history = pd.read_csv("../data/test_history_feature_B.csv", dtype={'link_ID':str})
    print(len(train_history))
    print(len(test_history))
    print(train_history["link_ID"].dtypes)
    print(test_history["link_ID"].dtypes)
    print("train and test history file load finished!")

    train2 = pd.merge(train1, train_history, on=["link_ID", "date", "hour"], how="inner")
    test2 = pd.merge(test1, test_history, on=["link_ID", "date", "hour"], how="left")

    print(len(train2))
    print(len(test2))

    train2.to_csv(outputTrainFile, index=False)
    test2.to_csv(outputTestFile, index=False)

if __name__ == "__main__":

    link_feature_filePath = "../data/link_feature.csv"
    train_time_feature_filePath = "../data/train_time_feature_B.csv"
    #train_time_feature_filePath = "../data/train_8-9_clock.csv"
    test_time_feature_filePath = "../data/test_time_feature_B.csv"
    output_train_filePath = "../data/trainB_1.csv"
    output_test_filePath = "../data/testB_1.csv"

    combine_Time_link_feature(link_feature_filePath, train_time_feature_filePath, test_time_feature_filePath,
                              output_train_filePath, output_test_filePath)

    inputTrainFile = "../data/trainB_1.csv"
    inputTestFile = "../data/testB_1.csv"
    output_train_filePath = "../data/trainB_2.csv"
    output_test_filePath = "../data/testB_2.csv"
    mix_history(inputTrainFile, inputTestFile, output_train_filePath, output_test_filePath)

