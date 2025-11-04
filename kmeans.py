import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv('Mall_Customers.csv')

x=data[['Annual Income (k$)','Spending Score (1-100)']]

model=KMeans(n_clusters=5,random_state=0)
model.fit(x)
y_pred=model.predict(x)

plt.scatter(x.iloc[y_pred==0,0],x.iloc[y_pred==0,1],s=50,c='red',label='cluster1')
plt.scatter(x.iloc[y_pred==1,0],x.iloc[y_pred==1,1],s=50,c='blue',label='cluster2')
plt.scatter(x.iloc[y_pred==2,0],x.iloc[y_pred==2,1],s=50,c='green',label='cluster3')
plt.scatter(x.iloc[y_pred==3,0],x.iloc[y_pred==3,1],s=50,c='cyan',label='cluster4')
plt.scatter(x.iloc[y_pred==4,0],x.iloc[y_pred==4,1],s=50,c='yellow',label='cluster5')

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=200,c='black',marker='x',label='centroid')
plt.legend()
plt.show()