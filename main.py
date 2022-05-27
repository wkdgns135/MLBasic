from sklearn.linear_model import LogisticRegression
import pandas as pd

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
df_test['Survived'] = pd.read_csv("gender_submission.csv")['Survived']

def data_processing(df):
    del df['Name']
    del df['Ticket']

    del df['Cabin']
    df['Age'].fillna((df['Age'].mean()),inplace=True)
    df.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)
    df = pd.get_dummies(df, drop_first=True)

    del df['PassengerId']
    del df['Age']
    del df['SibSp']
    del df['Parch']
    del df['Embarked_Q']

    return df

df_train = data_processing(df_train)
df_test = data_processing(df_test)

train_x = df_train.drop("Survived", axis=1)
train_y = df_train['Survived']

test_x = df_test.drop("Survived", axis=1)
test_y = df_test['Survived']

logistic  = LogisticRegression()
logistic.fit(train_x, train_y)
score = logistic.score(test_x, test_y)

print(score)