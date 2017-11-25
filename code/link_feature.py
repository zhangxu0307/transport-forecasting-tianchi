import pandas as pd
import numpy as np

linkInfoPath = "../data/gy_contest_link_info.txt"
topPath = "../data/gy_contest_link_top1.txt"

linkInfo = pd.read_csv(linkInfoPath, delimiter=";")
topInfo = pd.read_csv(topPath, delimiter=";")

linkFeature = pd.merge(linkInfo, topInfo, how="inner")
#print(linkFeature)

linkInfo = linkInfo.set_index("link_ID")
print(linkInfo)

# 上游特征
linkFeature["in_links_num"] = None
linkFeature["in_length_sum"] = None
linkFeature["in_length_diff"] = None
linkFeature["in_width_sum"] = None
linkFeature["in_width_diff"] = None

# 下游特征
linkFeature["out_links_num"] = None
linkFeature["out_length_sum"] = None
linkFeature["out_length_diff"] = None
linkFeature["out_width_sum"] = None
linkFeature["out_width_diff"] = None


for i in range(132):
    record = linkFeature.ix[i, :]

    # 上游道路特征，可增加
    linkIn = record["in_links"]
    if not isinstance(linkIn, str): # 无上游道路
        #record["in_links_num"] = 0
        linkFeature.ix[i, "in_links_num"] = 0
        linkFeature.ix[i, "in_length_sum"] = 0
        linkFeature.ix[i, "in_length_diff"] = 0
        linkFeature.ix[i, "in_width_sum"] = 0
        linkFeature.ix[i, "in_width_diff"] = 0
    else:
        inLinks = linkIn.split("#")
        in_lengthSum = 0
        in_wideSum = 0
        in_wideDiff = 0
        in_lengthDiff = 0
        for link in inLinks:
            link = int(link)
            in_lengthSum += linkInfo.ix[link, "length"]
            in_wideSum += linkInfo.ix[link, "width"]
        in_lengthDiff = record["length"]-in_lengthSum
        in_wideDiff = record["width"]-in_wideSum
        inNum = len(inLinks)
        print(in_lengthDiff, in_lengthSum, in_wideDiff, in_wideSum, inNum)
        linkFeature.ix[i, "in_links_num"] = inNum
        linkFeature.ix[i, "in_length_sum"] = in_lengthSum
        linkFeature.ix[i, "in_length_diff"] = in_lengthDiff
        linkFeature.ix[i, "in_width_sum"] = in_wideSum
        linkFeature.ix[i, "in_width_diff"] = in_wideDiff

    # 下游道路特征，可增加
    linkOut = record["out_links"]
    if not isinstance(linkOut, str):
        record["out_links_num"] = 0
        linkFeature.ix[i, "out_links_num"] = 0
        linkFeature.ix[i, "out_length_sum"] = 0
        linkFeature.ix[i, "out_length_diff"] = 0
        linkFeature.ix[i, "out_width_sum"] = 0
        linkFeature.ix[i, "out_width_diff"] = 0
    else:
        outLinks = linkOut.split("#")
        out_lengthSum = 0
        out_wideSum = 0
        out_wideDiff = 0
        out_lengthDiff = 0
        for link in outLinks:
            link = int(link)
            out_lengthSum += linkInfo.ix[link, "length"]
            out_wideSum += linkInfo.ix[link, "width"]
        out_lengthDiff = out_lengthSum- record["length"]
        out_wideDiff = out_wideSum - record["width"]
        outNum = len(outLinks)
        print(out_lengthDiff, out_lengthSum, out_wideDiff, out_wideSum, outNum)
        linkFeature.ix[i, "out_links_num"] = outNum
        linkFeature.ix[i, "out_length_sum"] = out_lengthSum
        linkFeature.ix[i, "out_length_diff"] = out_lengthDiff
        linkFeature.ix[i, "out_width_sum"] = out_wideSum
        linkFeature.ix[i, "out_width_diff"] = out_wideDiff

print(linkFeature.columns)
print(linkFeature)
linkFeature = linkFeature.drop(["in_links", "out_links"], axis=1) # 去掉原始数据中inlinks和outlinkes出现的空缺
linkFeature.to_csv("../data/link_feature.csv", index=False)



