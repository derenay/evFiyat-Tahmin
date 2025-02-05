import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV




# Load dataset
df = pd.read_csv("clear_data.csv")
df = df.drop(columns=df.columns[0])
# Select features and target
X = df.drop(['price', 'city', 'district', 'neighbourhood'], axis=1)
y = df['price']

# Scale numerical features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()  # Flatten to 1D

# Split dataset into train (80%), validation (10%), test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Data Preprocessing Completed ✅")

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = lr_model.predict(X_val)

# Evaluate performance
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"Linear Regression - Validation RMSE: {val_rmse:.2f}")
print(f"Linear Regression - Validation R² Score: {val_r2:.4f}")


# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred_rf = rf_model.predict(X_val)

# Evaluate performance
val_rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
val_r2_rf = r2_score(y_val, y_val_pred_rf)

print(f"Random Forest - Validation RMSE: {val_rmse_rf:.2f}")
print(f"Random Forest - Validation R² Score: {val_r2_rf:.4f}")



# Train Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred_gb = gb_model.predict(X_val)

# Evaluate performance
val_rmse_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_gb))
val_r2_gb = r2_score(y_val, y_val_pred_gb)

print(f"Gradient Boosting - Validation RMSE: {val_rmse_gb:.2f}")
print(f"Gradient Boosting - Validation R² Score: {val_r2_gb:.4f}")




# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Perform Grid Search
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_gb = grid_search.best_estimator_

# Predict on validation set
y_val_pred_best_gb = best_gb.predict(X_val)

# Evaluate performance
val_rmse_best_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_best_gb))
val_r2_best_gb = r2_score(y_val, y_val_pred_best_gb)

print(f"Best Gradient Boosting - Validation RMSE: {val_rmse_best_gb:.2f}")
print(f"Best Gradient Boosting - Validation R² Score: {val_r2_best_gb:.4f}")
print("Best Parameters:", grid_search.best_params_)



# Predict on test set
y_test_pred = best_gb.predict(X_test)

# Convert predictions back to original scale
y_test_pred_real = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Compute RMSE on real prices
test_rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
test_r2 = r2_score(y_test_real, y_test_pred_real)

print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R² Score: {test_r2:.4f}")


# Example new house features (replace with real values)
new_house = [[2, 1, 90, 1, 3]]  

# Scale new house features
new_house_scaled = scaler_X.transform(new_house)

# Predict price
predicted_price_scaled = best_gb.predict(new_house_scaled)

# Convert back to original scale
predicted_price_real = scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1))

print(f"Predicted House Price: {predicted_price_real[0][0]:,.2f}")



