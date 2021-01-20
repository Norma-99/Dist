from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('../datasets/Shuffle2.csv')
proc = df[["c","d","e","f"]].to_numpy()

for i in range(0,len(proc)):
    #ip treatment
    proc[i,0] = proc[i,0].split(".")
    proc[i,0] = proc[i,0][-1]
    proc[i,2] = proc[i,2].split(".")
    proc[i,2] = proc[i,2][-1]

    #0xs treatment
    if "0x" in proc[i,1]:
        proc[i,1] = proc[i,1].replace('0x','')
        proc[i,1] = int(proc[i,1], 16)
    if "0x" in proc[i,3]:
        proc[i,3] = proc[i,3].replace('0x','')
        proc[i,3] = int(proc[i,3], 16)

newCol0 = pd.Series(proc[:,0], name="c")
newCol1 = pd.Series(proc[:,1], name="d")
newCol2 = pd.Series(proc[:,2], name="e")
newCol3 = pd.Series(proc[:,3], name="f")

df.update(newCol0)
df.update(newCol1)
df.update(newCol2)
df.update(newCol3)

df["c"] = df["c"].astype('float64')
df["d"] = df["d"].astype('float64')
df["e"] = df["e"].astype('float64')
df["f"] = df["f"].astype('float64')

xdf = df.iloc[:,0:23]
xdf = pd.get_dummies(xdf) # 19358 - 80% 15486 - 20% 3872
x_traindf = xdf.iloc[:15486]
x_testdf = xdf.iloc[15486:]

ydf = df.iloc[:,23]
y_traindf = ydf.iloc[:15486]
y_testdf = ydf.iloc[15486:]

x_train = x_traindf.to_numpy()
x_train = StandardScaler().fit_transform(x_train)
x_test = x_testdf.to_numpy()
x_test = StandardScaler().fit_transform(x_test)

y_train = y_traindf.to_numpy()
y_test = y_testdf.to_numpy()

# Save information into .pickle format
validation_pair = x_test, y_test
training_pair = x_train, y_train
with open('../datasets/validation_dataset.pickle', 'wb') as f:
    pickle.dump(validation_pair, f)

with open('../datasets/train_dataset.pickle', 'wb') as f:
    pickle.dump(training_pair, f)

