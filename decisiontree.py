import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data=pd.read_csv('Buy_Computer.csv')

for col in data:
 le=LabelEncoder()
 data[col]=le.fit_transform(data[col])
 
x=data.drop(columns='Buy_Computer')
y=data['Buy_Computer']

tree=DecisionTreeClassifier()
tree.fit(x,y)

plt.figure(figsize=(12,8))
plot_tree(tree,
          feature_names=list(x.columns),
          class_names=['yes','no'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.show()


