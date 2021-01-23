from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('./datasets/Bot/Shuffle_category.csv')
df=df.fillna('0')
print(df.dtypes)
proc = df[["c","d","e","f"]].to_numpy()

for i in range(0,len(proc)):
    #ip treatment
    proc[i,0] = proc[i,0].split(".")
    proc[i,0] = proc[i,0][-1]
    proc[i,2] = proc[i,2].split(".")
    proc[i,2] = proc[i,2][-1]


newCol0 = pd.Series(proc[:,0], name="c")
newCol2 = pd.Series(proc[:,2], name="e")


df.update(newCol0)
df.update(newCol2)


df["c"] = df["c"].astype('float64')
df["d"] = df["d"].astype('float64')
df["e"] = df["e"].astype('float64')
df["f"] = df["f"].astype('float64')

xdf = df.iloc[:,0:24]
xdf = pd.get_dummies(xdf) # 19349 - 80% 15479 - 20% 3870
x_traindf = xdf.iloc[:15479]
x_testdf = xdf.iloc[15479:]

ydf = df.iloc[:,24]
y_traindf = ydf.iloc[:15479]
y_testdf = ydf.iloc[15479:]

x_train = x_traindf.to_numpy()
x_train = StandardScaler().fit_transform(x_train)
x_test = x_testdf.to_numpy()
x_test = StandardScaler().fit_transform(x_test)

y_train = y_traindf.to_numpy()
y_test = y_testdf.to_numpy()

# Save information into .pickle format
validation_pair = x_test, y_test
training_pair = x_train, y_train
with open('./datasets/Bot/validation_dataset.pickle', 'wb') as f:
    pickle.dump(validation_pair, f)

with open('./datasets/Bot/train_dataset.pickle', 'wb') as f:
    pickle.dump(training_pair, f)
