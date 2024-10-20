import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv("Dataset_Unsupervised/Unsupervised_Dataset.csv", low_memory=False)
print(data.head())
print(data.info())

# Impute missing values with mean for specified columns
columns_to_impute = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes'
]

# Fill missing values with mean for specified columns
for column in columns_to_impute:
    if column in data.columns:
        data[column] = data[column].fillna(data[column].mean())

# Identify non-numeric columns and convert them to numeric if necessary
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns before encoding:", non_numeric_columns)

# Convert non-numeric columns to numeric using Label Encoding
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, forcing errors to NaN

# Handle NaN values in encoded columns (if any)
data.fillna(0, inplace=True)  # You can choose a different strategy for filling NaNs

print("Data types after processing:")
print(data.dtypes)

# Correlation analysis and visualization
numeric_data = data.select_dtypes(include=[np.number])  # This will include only numeric columns
corr_matrix = numeric_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Network Flow Features')
plt.show()

# Step 1: Define features for clustering
features = numeric_data.drop(columns=['Class'])  # Exclude the Class column for clustering

# Step 2: Determine Optimal Number of Clusters using Elbow Method and Silhouette Analysis

# Elbow Method
inertia = []
silhouette_scores = []
k_values = range(2, 11)  # Testing k from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)

    inertia.append(kmeans.inertia_)

    # Calculate silhouette score only if k > 1
    if k > 1:
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)

# Plotting Elbow Method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)

# Plotting Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_values[1:], silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values[1:])
plt.tight_layout()
plt.show()

# Choose optimal k based on visual inspection from the plots above.
optimal_k = int(input("Enter the optimal number of clusters based on the plots: "))

# Step 3: Apply K-Means Clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(features)

# Add cluster labels to the original dataframe
data['Cluster'] = clusters

# Step 4: Evaluate Clustering Performance using Silhouette Score for optimal k
silhouette_avg = silhouette_score(features, clusters)
print(f'Silhouette Score for optimal k ({optimal_k}): {silhouette_avg}')

# Step 5: Visualize Clusters (Optional)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], hue=data['Cluster'], palette='viridis', alpha=0.6)
plt.title('K-Means Clustering Results')
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.legend(title='Cluster')
plt.show()