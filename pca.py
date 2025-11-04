import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
iris = pd.read_csv('Iris.csv')
X = iris.drop(columns='Species')
y = iris['Species']

# Encode species to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualize 2D PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.show()

# Algorithm Steps:

# Step 1: Import required libraries (e.g., pandas, numpy, sklearn.decomposition.PCA).
# Step 2: Load the dataset and standardize the features using StandardScaler() (mean = 0, variance = 1).
# Step 3: Compute the covariance matrix (Σ) of the standardized data.

# Step 4: Calculate eigenvalues and eigenvectors of the covariance matrix.
# Step 5: Sort the eigenvectors by decreasing eigenvalues (most variance first).
# Step 6: Choose the top k eigenvectors (corresponding to largest eigenvalues).
# Step 7: Form a projection matrix (W) using selected eigenvectors.
# Step 8: Transform the original data:

