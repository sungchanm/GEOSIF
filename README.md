# GEOSIF (https://doi.org/10.1016/j.rse.2024.114284)
GEOSIF: A continental-scale sub-daily reconstructed solar-induced fluorescence derived from OCO-3 and GK-2A over Eastern Asia and Oceania (Jeong et al., 2024).

This repository contains code for training an XGBoost regression model to predict SIF using four-band spectral reflectance, shortwave radiation, and vapor pressure deficit. 
It also includes functionality for hyperparameter tuning and SHAP analysis to interpret model predictions.

# Requirements

- Python 3.7 or higher
- NumPy
- pandas
- xgboost
- scikit-learn
- scipy
- shap
- matplotlib

# Note
This study used BRDF-normalized GK-2A Blue, Green, Red, and NIR reflectance. Please consider the different processing levels of reflectance datasets to apply our approach to other GEO satellites (e.g. Himawari or GOES).

