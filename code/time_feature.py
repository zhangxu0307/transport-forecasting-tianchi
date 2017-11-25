import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

def getDate(dateInput): # 获取日期

    date = datetime.strptime(dateInput, "%Y-%m-%d")
    return date

def getMonth(dateInput): # 获取月份

    #dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    month = dateInput.month
    return month

def getDay(dateInput): # 获取天

    #dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    day = dateInput.day
    return day

def getWeekDay(dateInput): # 获取星期几

    #dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    return dateInput.isoweekday()

def isWeekend(weekday): # 是否为周末

    if weekday == 6 or weekday == 7:
        return 1
    else:
        return 0

# 获取清明、端午、五一三个节假日
def getHoliday(dateInput):

    month = dateInput.month
    day = dateInput.day

    if (month == 4 and day == 2) or  (month == 4 and day == 3) or (month == 4 and day == 4) or\
            (month == 4 and day == 30) or (month == 5 and day == 1) or (month == 5 and day == 2) or\
        (month == 6 and day == 9) or (month == 6 and day == 10) or (month == 6 and day == 11):
        return 1
    else:
        return 0

def getStartDateTime(timeInput): # 获取开始的日期时间
    dateTime = datetime.strptime(timeInput.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S")
    return dateTime


def getHour(dateTime): # 获取开始的小时

    #dateTime = datetime.strptime(timeInput.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S")
    hour = dateTime.hour
    return hour

def getMinute(dateTime): # 获取开始的分钟

    #dateTime = datetime.strptime(timeInput.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S")
    minute = dateTime.minute
    return minute

def getMoringPeak(hour): # 是否是早高峰

    if hour >= 8 and hour <= 10: # 8-10点早高峰
        return 1
    else:
        return 0

def getEveningPeak(hour): # 是否是晚高峰

    if hour >= 17 and hour <= 19: # 17-19点晚高峰
        return 1
    else:
        return 0

def trainTimeFeature():

    trainPath = "../data/gy_contest_traveltime_training_data_second.txt"
    train = pd.read_csv(trainPath, delimiter=";")

    # 获取日期信息
    train["date"] = train["date"].apply(getDate)
    train["month"] = train["date"].apply(getMonth)
    train["day"] = train["date"].apply(getDay)
    train["weekday"] = train["date"].apply(getWeekDay)
    train["is_weekend"] = train["weekday"].apply(isWeekend)
    #train["holiday"] = train["date"].apply(getHoliday)

    # 获取时间信息
    train["start_date_time"] = train["time_interval"].apply(getStartDateTime)
    train["hour"] = train["start_date_time"].apply(getHour)
    train["minute"] = train["start_date_time"].apply(getMinute)

    # 获取早晚高峰信息
    train["morning_peak"] = train["hour"].apply(getMoringPeak)
    train["evening_peak"] = train["hour"].apply(getEveningPeak)
    print(len(train))

    train.to_csv("../data/train_time_feature_B.csv", index=False)

def testTimeFeature():

    testPath = "../data/gy_contest_result_template_second.txt"
    test = pd.read_csv(testPath, delimiter="#", header=0)

    #test.columns = ["link_ID", "date","time_interval", "travel_time"]

    # 获取日期信息
    test["date"] = test["date"].apply(getDate)
    test["month"] = test["date"].apply(getMonth)
    test["day"] = test["date"].apply(getDay)
    test["weekday"] = test["date"].apply(getWeekDay)
    test["is_weekend"] = test["weekday"].apply(isWeekend)
    #test["holiday"] = test["date"].apply(getHoliday)

    # 获取时间信息
    test["start_date_time"] = test["time_interval"].apply(getStartDateTime)
    test["hour"] = test["start_date_time"].apply(getHour)
    test["minute"] = test["start_date_time"].apply(getMinute)

    # 获取早晚高峰信息
    test["morning_peak"] = test["hour"].apply(getMoringPeak)
    test["evening_peak"] = test["hour"].apply(getEveningPeak)
    print(len(test))

    test.to_csv("../data/test_time_feature_B.csv", index=False)



if __name__ == "__main__":

    trainTimeFeature()

    testTimeFeature()






