import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

def getMonth(dateInput):

    dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    month = dateTime.month
    return month

def getDay(dateInput):

    dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    day = dateTime.day
    return day

def isWeekend(dateInput):

    dateTime = datetime.strptime(dateInput, "%Y-%m-%d")
    return dateTime.isoweekday()


def getHour(timeInput):

    dateTime = datetime.strptime(timeInput.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S")
    hour = dateTime.hour
    return hour

def getMin(timeInput):

    dateTime = datetime.strptime(timeInput.lstrip("[").split(",")[0], "%Y-%m-%d %H:%M:%S")
    minute = dateTime.minute
    return minute

def getMoringPeak(hour):

    if hour >= 8 and hour <= 10: # 8-10点早高峰
        return 1
    else:
        return 0

def getEveningPeak(hour):

    if hour >= 17 and hour <= 19: # 17-19点晚高峰
        return 1
    else:
        return 0


if __name__ == "__main__":

    trainPath = "../data/gy_contest_link_traveltime_training_data.txt"
    train = pd.read_csv(trainPath, delimiter=";")

    train["month"] = train["date"].apply(getMonth)
    train["day"] = train["date"].apply(getDay)
    train["weekday"] = train["date"].apply(isWeekend)

    train["hour"] = train["time_interval"].apply(getHour)
    train["min"] = train["time_interval"].apply(getMin)

    train["morning_peak"] = train["hour"].apply(getMoringPeak)
    train["evening_peak"] = train["hour"].apply(getEveningPeak)
    print(train.head(10))


    train.to_csv("../data/train_feature.csv", index=False)




