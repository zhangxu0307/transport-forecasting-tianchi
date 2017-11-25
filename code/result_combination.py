import pandas as pd
import numpy as np


result1 = pd.read_csv("../result/result7_0338.txt", delimiter="#", header=-1)
result2 = pd.read_csv("../result/result12_0335.txt", delimiter="#", header=-1)
result1.columns = ["link_ID", "date","time_interval", "travel_time"]
result2.columns = ["link_ID", "date","time_interval", "travel_time"]
print(len(result1))
print(len(result2))
#finalRes = pd.read_csv("./data/result_combination.csv")

#print((result1["volume"] == finalRes["volume"]))

sumWeight = (1.0-0.338)+(1.0-0.335)

result1["travel_time"] = result1["travel_time"]*(1.0-0.338)/sumWeight+result2["travel_time"]*(1.0-0.335)/sumWeight

result1.to_csv("../result/result_combination1.txt", index=False, sep="#", header=False)




