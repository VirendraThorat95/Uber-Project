#!/usr/bin/env python
# coding: utf-8

# # Traffic Congestion Forecasting
# # Model Development, Evaluation & Cross-Validation
# 
# # ----------------------------
# # 1. Import Libraries
# # ----------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from xgboost import XGBRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 
# # ----------------------------
# # 2. Load Dataset
# # ----------------------------
# data_path = r"C:\Desktop\Upgrad_Python\Uber Project\Data.csv"
# df = pd.read_csv(data_path)
# 
# # Convert DateTime column to datetime type
# df['DateTime'] = pd.to_datetime(df['DateTime'])
# 
# # Quick look at the dataset
# print(df.head())
# 
# # ----------------------------
# # 3. Feature Engineering
# # ----------------------------
# # Time-based features
# df['Hour'] = df['DateTime'].dt.hour
# df['DayOfWeek'] = df['DateTime'].dt.dayofweek
# df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
# 
# # Lag features
# for lag in range(1, 4):
#     df[f'Vehicles_lag{lag}'] = df.groupby('Junction')['Vehicles'].shift(lag)
# 
# # Rolling mean of past 3 time steps per junction
# df['Vehicles_roll3'] = (
#     df.groupby('Junction')['Vehicles']
#       .rolling(3)
#       .mean()
#       .reset_index(level=0, drop=True)
# )
# 
# # Drop rows with NaN (from lag/rolling calculations)
# df = df.dropna()
# 
# # ----------------------------
# # 4. Train-Validation Split
# # ----------------------------
# train_size = int(len(df) * 0.8)
# train_df = df.iloc[:train_size]
# val_df = df.iloc[train_size:]
# 
# features = [
#     'Hour', 'DayOfWeek', 'IsWeekend',
#     'prcp', 'wspd',
#     'Vehicles_lag1', 'Vehicles_lag2', 'Vehicles_lag3',
#     'Vehicles_roll3'
# ]
# 
# X_train = train_df[features]
# y_train = train_df['Vehicles']
# X_val = val_df[features]
# y_val = val_df['Vehicles']
# 
# # ----------------------------
# # 5. Model Training
# # ----------------------------
# model = XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     random_state=42
# )
# model.fit(X_train, y_train)
# 
# # ----------------------------
# # 6. Validation & Metrics
# # ----------------------------
# y_pred = model.predict(X_val)
# 
# mae = mean_absolute_error(y_val, y_pred)
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# r2 = r2_score(y_val, y_pred)
# 
# print("Model Performance on Validation Set")
# print(f"MAE : {mae:.2f}")
# print(f"RMSE : {rmse:.2f}")
# print(f"R² : {r2:.3f}")
# 
# # Plot Actual vs Predicted
# plt.figure(figsize=(10, 4))
# plt.plot(y_val.values[:200], label='Actual')
# plt.plot(y_pred[:200], label='Predicted')
# plt.title("Predicted vs Actual Traffic")
# plt.xlabel("Time Index")
# plt.ylabel("Vehicle Count")
# plt.legend()
# plt.show()
# 
# # ----------------------------
# # 7. Time Series Cross-Validation
# # ----------------------------
# tscv = TimeSeriesSplit(n_splits=5)
# cv_results = []
# 
# for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
#     X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
# 
#     model.fit(X_tr, y_tr)
#     preds = model.predict(X_va)
# 
#     cv_results.append({
#         'Fold': fold,
#         'MAE': mean_absolute_error(y_va, preds),
#         'RMSE': np.sqrt(mean_squared_error(y_va, preds)),
#         'R2': r2_score(y_va, preds)
#     })
# 
# cv_df = pd.DataFrame(cv_results)
# 
# print("\nCross-Validation Results per Fold:")
# print(cv_df)
# 
# print("\nAverage CV Metrics:")
# print(cv_df[['MAE', 'RMSE', 'R2']].mean())
# 

# 

# In[ ]:
input("\nPress Enter to exit...")





