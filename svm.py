import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data=pd.read_csv('Iris.csv')
x=data[['SepalLengthCm','PetalLengthCm']]
y=data['Species']

le=LabelEncoder()
y_new=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y_new,test_size=0.3,random_state=42)

sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.transform(x_test)

model=SVC(kernel='linear')
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

cm=confusion_matrix(y_test,y_pred)
dis=ConfusionMatrixDisplay(confusion_matrix=cm)
dis.plot(cmap='Blues')
plt.show()

from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    model, x_train_scaled, response_method="predict",
    cmap='coolwarm', alpha=0.3
)

# Plot data points (also scaled)
plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1],
            c=y_train, edgecolors='k', cmap='coolwarm')
plt.title("SVM Decision Boundary (Auto)")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Petal Length (scaled)")
plt.show()
