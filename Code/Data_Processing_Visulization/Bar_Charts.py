#Bar_Charts

import pandas as pd
import os 
import glob
data = []
result = glob.glob( '../../Result/CSV/*.csv' )
for x in result:
	df = pd.read_csv(x).sort_values('AUC',ascending=False).head(1)
	data.append(df)
data = pd.concat(data, axis=0)
data.index =data.ALG_NAME
print(data)
data.to_csv("../../Result/CSV/RESULT.csv",index=False)
import matplotlib.pyplot as plt

# data[["AUC", "Average F-Score"]].plot.bar()
# plt.show()



# data[["PRESCISION", "RECALL",'ACCURACY']].plot.bar()
# plt.show()



# data[['RUNTIME']].plot.bar()
# plt.show()

