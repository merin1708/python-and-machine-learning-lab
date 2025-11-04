import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score

data=pd.read_csv('loan.csv')

data=data.dropna()
data=data.drop(columns='Loan_ID')


le=LabelEncoder()
data['Loan_Status']=le.fit_transform(data['Loan_Status'])

x=pd.get_dummies(data.drop(columns='Loan_Status'),drop_first=True)
y=data['Loan_Status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy score:",accuracy_score(y_test,y_pred))
print("precision score:",precision_score(y_test,y_pred))
print("Recall score:",recall_score(y_test,y_pred))
