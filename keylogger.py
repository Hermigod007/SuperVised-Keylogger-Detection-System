import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier  # Import XGBoost Classifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.preprocessing import StandardScaler

# Load and Inspect the Data
data = pd.read_csv('Datasets/Dataset.csv', low_memory=False)
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

# Encode the Class variable
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# Identify non-numeric columns and convert them to numeric if necessary
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns before encoding:", non_numeric_columns)

# Convert non-numeric columns to numeric using Label Encoding
for col in non_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, forcing errors to NaN

# Handle NaN values in encoded columns (if any)
data.fillna(0, inplace=True)  # You can choose a different strategy for filling NaNs

# Now all columns should be numeric; let's check again
print("Data types after processing:")
print(data.dtypes)

# Correlation analysis and visualization
numeric_data = data.select_dtypes(include=[np.number])  # This will include only numeric columns
corr_matrix = numeric_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)  # Set annot=True to show correlation values
plt.title('Correlation Matrix of Network Flow Features')
plt.show()

# Training and Testing of Data (using all features)
X = data.drop('Class', axis=1)  # Features (all columns except Class)
y = data['Class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model with increased max_iter and scaling
logistic_model = LogisticRegression(max_iter=10000)

# Fit the model to the scaled training data
logistic_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Calculate accuracy and print results for Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'Accuracy of Logistic Regression: {accuracy_logistic*100:.2f}')

# Confusion matrix visualization for Logistic Regression
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logistic, annot=True, fmt='d', cmap='Reds', xticklabels=['Benign', 'Keylogger'], yticklabels=['Benign', 'Keylogger'])
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data using Decision Tree
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set using Decision Tree
y_pred_tree = decision_tree_model.predict(X_test)

# Calculate accuracy and print results for Decision Tree
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f'Accuracy of Decision Tree: {accuracy_tree*100:.2f}')

# Confusion matrix visualization for Decision Tree
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Keylogger'], yticklabels=['Benign', 'Keylogger'])
plt.title('Confusion Matrix for Decision Tree')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)

# Fit the model to the training data using Random Forest
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set using Random Forest
y_pred_rf = random_forest_model.predict(X_test)

# Calculate accuracy and print results for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy of Random Forest: {accuracy_rf*100:.2f}')

# Confusion matrix visualization for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['Benign', 'Keylogger'], yticklabels=['Benign', 'Keylogger'])
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Initialize the Gradient Boosting model (XGBoost)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Fit the model to the training data using XGBoost (Gradient Boosting)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Calculate accuracy and print results for XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'Accuracy of XGBoost: {accuracy_xgb*100:.2f}')

# Confusion matrix visualization for XGBoost
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Purples', xticklabels=['Benign', 'Keylogger'], yticklabels=['Benign', 'Keylogger'])
plt.title('Confusion Matrix for XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
