import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure settings
pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# Read data
energy = pd.read_csv("./energy.csv", parse_dates=['timestamp'], index_col='timestamp')

# Create training and testing data sets
train_start_dt = '2014-11-01 00:00:00'
test_start_dt = '2014-12-30 00:00:00'

train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
test = energy.copy()[energy.index >= test_start_dt][['load']]
print(train.shape, test.shape)

# Scale data
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
test['load'] = scaler.transform(test)

# Model parameters
n_test = len(test)
order = (2, 1, 0)
seasonal_order = (1, 1, 1, 24)

# Initialize variables
history = [x for x in train['load']]

model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)

try:
    print(f"Fitting model")
    model_fit = model.fit()
except Exception as e:
    print(f"Error fitting model at timestamp {test.index[t]}: {e}")

print("Forecasting...")
predictions = model_fit.forecast(steps=n_test)
print(predictions)
print(predictions.shape)

eval_df = pd.DataFrame(predictions, columns=["prediction"])
eval_df['timestamp'] = test.index
eval_df['actual'] = np.array(np.transpose(test)).ravel()

# Calculate MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

print('MAPE: ', mape(eval_df['actual'], eval_df['prediction']), '%')

eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))
plt.show()
