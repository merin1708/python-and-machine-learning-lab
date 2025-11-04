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


# Algorithm Steps:

# Step 1: Import necessary libraries (e.g., pandas, matplotlib, sklearn.cluster.KMeans).
# Step 2: Load the dataset into a variable (e.g., X for feature data).
# Step 3: Select the number of clusters K to form (decided manually or by Elbow method).
# Step 4: Initialize K cluster centroids randomly.
# Step 5: For each data point, assign it to the nearest centroid using Euclidean distance.
# Step 6: After all points are assigned, recompute each centroid as the mean of the points in that cluster.
# Step 7: Repeat Steps 5–6 until the centroids no longer change (convergence).
# Step 8: Display the cluster assignments and visualize them using scatter plots.
# Step 9: Evaluate performance using WCSS (Within Cluster Sum of Squares) or Elbow Method if needed.
