# -*- coding: utf-8 -*-
"""train_model(TMDB).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14cKnyCj0-4u1MtGNe4E6kh19V9ihQcvf
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# Load data
df = pd.read_csv('/content/movies_data.csv')

# Preprocess the data
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(df['genres'])

# Prepare features and target variable
X = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_)
X['budget'] = df['budget']
X['runtime'] = df['runtime']

y = df['revenue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and preprocessing objects
with open('/content/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('/content/mlb_genres.pkl', 'wb') as mlb_file:
    pickle.dump(mlb_genres, mlb_file)

# Save feature names used in training
feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as feature_file:
    pickle.dump(feature_names, feature_file)

# Save the dataset for Streamlit use
df.to_pickle('/content/all_movies.pkl')

df = pd.read_pickle('all_movies.pkl')
print("Columns in the DataFrame:", df.columns)

"""### **Accuracy**"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

"""### **Visualizing model**"""

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.show()

"""### **Randomforestregresssor**"""

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Random Forest - Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"Random Forest - R² Score: {r2_rf:.2f}")

# Save the RandomForest model
with open('/content/rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

import sklearn
print(sklearn.__version__)

"""### **Hyperparameter tuning**"""

from sklearn.model_selection import GridSearchCV

# Define the model
rf_model = RandomForestRegressor(random_state=42)

# Define hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and model
print(f"Best parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred_best = best_rf_model.predict(X_test)

# Calculate metrics
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Optimized Random Forest - Mean Absolute Error (MAE): {mae_best:.2f}")
print(f"Optimized Random Forest - Mean Squared Error (MSE): {mse_best:.2f}")
print(f"Optimized Random Forest - Root Mean Squared Error (RMSE): {rmse_best:.2f}")
print(f"Optimized Random Forest - R² Score: {r2_best:.2f}")

"""### **Gradient boosting**"""

from sklearn.ensemble import GradientBoostingRegressor

# Initialize and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Calculate metrics
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting - Mean Absolute Error (MAE): {mae_gb:.2f}")
print(f"Gradient Boosting - Mean Squared Error (MSE): {mse_gb:.2f}")
print(f"Gradient Boosting - Root Mean Squared Error (RMSE): {rmse_gb:.2f}")
print(f"Gradient Boosting - R² Score: {r2_gb:.2f}")

"""### **Hyperparameter tuning for gradientboosting**"""

# Define the model
gb_model = GradientBoostingRegressor(random_state=42)

# Define hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize GridSearchCV
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid,
                              cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search_gb.fit(X_train, y_train)

# Best parameters and model
print(f"Best parameters: {grid_search_gb.best_params_}")
best_gb_model = grid_search_gb.best_estimator_

# Predict on the test set
y_pred_best_gb = best_gb_model.predict(X_test)

# Calculate metrics
mae_best_gb = mean_absolute_error(y_test, y_pred_best_gb)
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)
rmse_best_gb = np.sqrt(mse_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)

print(f"Optimized Gradient Boosting - Mean Absolute Error (MAE): {mae_best_gb:.2f}")
print(f"Optimized Gradient Boosting - Mean Squared Error (MSE): {mse_best_gb:.2f}")
print(f"Optimized Gradient Boosting - Root Mean Squared Error (RMSE): {rmse_best_gb:.2f}")
print(f"Optimized Gradient Boosting - R² Score: {r2_best_gb:.2f}")

# Save the model and preprocessing objects
with open('/content/best_gb_model.pkl', 'wb') as model_file:
    pickle.dump(grid_search_gb.best_estimator_, model_file)
