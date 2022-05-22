from sklearn.linear_model import LogisticRegression
import pandas as pd

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

logistic  = LogisticRegression()
# logistic.fit(train)