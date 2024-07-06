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

# Scale data
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
test['load'] = scaler.transform(test)

# Model parameters
HORIZON = 3
order = (2, 1, 0)
seasonal_order = (1, 1, 1, 24)

# Initialize variables
history = [x for x in train['load']]
predictions = []

# Forecasting loop
for t in range(test.shape[0]):
    model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
    
    try:
        model_fit = model.fit()
    except Exception as e:
        print(f"Error fitting model at timestamp {test.index[t]}: {e}")
        continue
    
    yhat = model_fit.forecast(steps=HORIZON)
    predictions.append(yhat)
    
    obs = list(test.iloc[t])
    history.append(obs[0])
    history.pop(0)
    
    print(test.index[t])
    print(t + 1, ': predicted =', yhat, 'expected =', obs)

# Prepare evaluation dataframe
eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
eval_df['actual'] = np.array(np.transpose(test)).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])

# Calculate MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

print('One step forecast MAPE: ', mape(eval_df[eval_df['h'] == 't+1']['actual'], eval_df[eval_df['h'] == 't+1']['prediction']), '%')
print('Multi-step forecast MAPE: ', mape(eval_df['actual'], eval_df['prediction']), '%')

# Plot predictions vs actuals
if HORIZON == 1:
    eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))
else:
    plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
    for t in range(1, HORIZON+1):
        plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0, label='Actual')
    for t in range(1, HORIZON+1):
        ax.plot(plot_df['timestamp'], plot_df['t+'+str(t)], color='blue', linewidth=4*math.pow(0.9, t), alpha=math.pow(0.8, t), label='t+'+str(t))
    
    ax.legend(loc='best')
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
