# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib


df = pd.read_csv("/Users/shahzaibhussain/Documents/AER 850 Project 1/Project_1_Data.csv")
print(df.info())

# Step 1
# ----------------------------------------------------------------------------
# Load the CSV file into a dataframe
df = pd.read_csv("/Users/shahzaibhussain/Documents/AER 850 Project 1/Project_1_Data.csv")

# Display the first few rows of the dataframe
print(df.head())


# Step 2
# ----------------------------------------------------------------------------
print(df.describe())
# Plot histograms for X, Y, and Z coordinates for each respective distributions
plt.figure(figsize=(14, 4))

# Plot X coordinate
plt.subplot(1, 3, 1)
plt.hist(df['X'], bins=15, color='green', edgecolor='black')
plt.title('Distribution in X Coordinate')

# Plot Y coordinate
plt.subplot(1, 3, 2)
plt.hist(df['Y'], bins=15, color='gray', edgecolor='black')
plt.title('Distribution in Y Coordinate')

# Plot Z coordinate
plt.subplot(1, 3, 3)
plt.hist(df['Z'], bins=15, color='blue', edgecolor='black')
plt.title('Distribution in Z Coordinate')

plt.tight_layout()
plt.show()

# Create a figure for the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot
sc = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis')

# Set axis labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of Coordinates')

# Add a color bar representing step
cbar = plt.colorbar(sc, ax=ax, label='Step')

# Show the plot
plt.show()

# Step 3
# ----------------------------------------------------------------------------
# Calculating the correlation between the coordinates X,Y,Z and the Step

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set the figure size
plt.figure(figsize=(8, 6))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.25)

# Add title for the plot
plt.title('Correlation Matrix of Coordinates and Step')

# Show the plot
plt.show()

# Step 4
# ----------------------------------------------------------------------------
# Prepare the target variable and set
X = df[['X', 'Y', 'Z']]
y = df['Step']

# Split the data for training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)
svc = SVC()

# Define Hyperparameter for each model
param_grid_log_reg = {'C': [0.1, 1, 10]}
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Perform GridSearch for Logistic Regression
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)
grid_search_log_reg.fit(X_train, y_train)

# Perform GridSearch for Random Forest
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Perform GridSearch for SVC
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5)
grid_search_svc.fit(X_train, y_train)

# RandomizedSearchCV for Random Forest
random_search_rf = RandomizedSearchCV(random_forest, param_distributions=param_grid_rf, n_iter=5, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

# Evaluate performance of the models on the testing set
y_pred_log_reg = grid_search_log_reg.predict(X_test)
y_pred_rf = random_search_rf.predict(X_test)
y_pred_svc = grid_search_svc.predict(X_test)

# Calculate accuracy
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svc = accuracy_score(y_test, y_pred_svc)

# Print the accuracy results from the calculated above
print(f'Logistic Regression Accuracy: {acc_log_reg:.4f}')
print(f'Random Forest Accuracy: {acc_rf:.4f}')
print(f'SVC Accuracy: {acc_svc:.4f}')

# Step 5: Model Performance Analysis
# ----------------------------------------------------------------------------
# Evaluate the Logistic Regression model
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_log_reg))

# Evaluate the Random Forest model
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# Evaluate the SVC model
print("SVC Classification Report")
print(classification_report(y_test, y_pred_svc))

# Creating a confusion matrix for the best performing model (assumed Random Forest)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', cbar=False)

# Add titles and labels for the plot
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Step 6: Performance Analysis
# ----------------------------------------------------------------------------
# Define the estimators for the stacking classifier
estimators = [
    ('rf', grid_search_rf.best_estimator_),  # Best Random Forest model from Grid Search
    ('svc', grid_search_svc.best_estimator_)  # Best SVC model from Grid Search
]

# Define the Stacking Classifier
stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacked model on the training set
stacked_model.fit(X_train, y_train)

# Making predictions with the stacked model on the test set
y_pred_stacked = stacked_model.predict(X_test)

# Evaluated the stacked model
print("Stacked Model Classification Report")
print(classification_report(y_test, y_pred_stacked))

# Generating the confusion matrix for stacked model
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Greens', cbar=False)

# Adding titles and labels
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Step 7: Model Evaluation - Save the trained stacked model
# -----------------------------------------------------------------------------
# Define the filename for saving the model
model_filename = 'stacked_model.joblib'

# Save the model
joblib.dump(stacked_model, model_filename)
print("Model saved as {model_filename}")

# Loading the saved model
loaded_model = joblib.load(model_filename)

# Example: New coordinate data for predictions
new_coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# Predict the steps for these new coordinates
predictions = loaded_model.predict(new_coordinates)

# Output the predictions
print("Predicted Maintenance Steps for the new coordinates:", predictions)

