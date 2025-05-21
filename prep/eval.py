import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import load_model

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def eval(model, true, stride, train_start):
    i = train_start
    pred = np.zeros(true.shape)
    pred[:train_start] = true[:train_start]
    while i < len(true):
        pred_diff = model.forecast(steps=stride)
        pred_diff = np.concatenate(([true[i-1]], pred_diff))
        pred[i-1:i+stride] = np.cumsum(pred_diff)
        model.append(true[i:i+stride])
        i += stride
    return pred


root_path = os.path.dirname("./datasets/")
set_name  = 'ETTh1'
data_path = set_name + '.csv'
dim_use = 7
train_start = 1024
test_span = 64
test_end = 4096
dim = 3

df_raw = pd.read_csv(os.path.join(root_path, data_path))

x = df_raw.iloc[:test_end, 3].values

model = load_model(f"m_{set_name}_{dim}.pkl")
y_seq = eval(model, x, test_span, train_start)

plt.figure(figsize=(20, 4))
plt.plot(x, label='original')
plt.plot(y_seq, label='predicted')
plt.legend()
plt.show()

print(MAE(x[train_start:test_end], y_seq[train_start:test_end]))
print(MSE(x[train_start:test_end], y_seq[train_start:test_end]))
