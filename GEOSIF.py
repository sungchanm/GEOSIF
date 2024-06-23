import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.io import loadmat
import shap
import matplotlib.pyplot as plt

# Load data from .mat files
input_sif = loadmat('geo_samples.mat')['all_sif']
input_air = loadmat('geo_samples.mat')['all_air']
input_vpd = loadmat('geo_samples.mat')['all_vpd'] / 100
input_rad = loadmat('geo_samples.mat')['all_rad']

input_red = loadmat('geo_samples.mat')['all_red']
input_nir = loadmat('geo_samples.mat')['all_nir']
input_blu = loadmat('geo_samples.mat')['all_blu']
input_gre = loadmat('geo_samples.mat')['all_gre']
input_lc = loadmat('geo_samples.mat')['all_lc']

# Stack all features together
all_features = np.hstack([input_sif, input_air, input_vpd, input_rad, input_nir, input_red, input_gre, input_blu, input_lc])

# Remove rows with NaN values
all_nan = np.sum(np.isnan(all_features), axis=1)
input_sif = input_sif[all_nan == 0].ravel()
input_vpd = input_vpd[all_nan == 0].ravel()
input_rad = input_rad[all_nan == 0].ravel()
input_red = input_red[all_nan == 0].ravel()
input_nir = input_nir[all_nan == 0].ravel()
input_gre = input_gre[all_nan == 0].ravel()
input_blu = input_blu[all_nan == 0].ravel()
input_lc = input_lc[all_nan == 0].ravel()

# Create DataFrame for the input features
input_gk2a = pd.DataFrame(np.round(np.column_stack([input_nir, input_red, input_gre, input_blu, input_rad, input_vpd]), 5),
                          columns=['nir', 'red', 'gre', 'blu', 'rad', 'vpd'])
input_oco3 = pd.Series(np.round(input_sif, 5))

# Perform stratified sampling
X_train, X_test, y_train, y_test = train_test_split(input_gk2a, input_oco3, test_size=0.3, stratify=input_lc, random_state=40)

# Define the hyperparameter grid
param_distributions = {
    'learning_rate': np.linspace(0.01, 0.3, 30),   # Learning rates from 0.01 to 0.3
    'max_depth': list(range(3, 13)),              # Max depths from 3 to 12
    'n_estimators': list(range(50, 201)),         # Number of trees from 50 to 200
    'subsample': np.linspace(0.5, 1, 51)          # Subsample ratios from 0.5 to 1
}

# Initialize the base model
base_model = xgb.XGBRegressor()

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_distributions,
    n_iter=100,    # Number of parameter settings that are sampled
    cv=5,          # 5-fold cross-validation
    verbose=1,
    random_state=40,
    n_jobs=-1      # Use all available cores
)

# Fit the model
random_search.fit(X_train, y_train)

# Best hyperparameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Predict with the best model
gk2a_sif = best_model.predict(input_gk2a)

print(f"Best hyperparameters: {best_params}")

# SHAP explanation
explainer = shap.Explainer(best_model)
shap_values = explainer(input_gk2a)

# Function to visualize SHAP values
def plot_shap_summary(shap_values, features):
    plt.figure()
    shap.summary_plot(shap_values, features)
    plt.title("SHAP Summary Plot for XGBoost Model")
    plt.show()

# Plot SHAP summary
plot_shap_summary(shap_values, input_gk2a)
