import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model import save_model

def adf_test(x_seq):
    result = adfuller(x_seq)
    return np.abs(result[1]) < 0.05  # p-value

def find_d(x_seq, fix_d=False):
    if fix_d:
        return np.diff(x_seq), 1, [x_seq[0]]
    d = 0
    init = []
    while not adf_test(x_seq):
        init.append(x_seq[0])
        x_seq = np.diff(x_seq)
        d += 1
    return x_seq, d, init

def reverse_diff(x_diff, d, init):
    while d > 0:
        d -= 1
        x_diff = np.concatenate(([init[d]], x_diff))
        x_diff = np.cumsum(x_diff)
    return x_diff

def vis_p_acf(x_seg):
    fig, axes = plt.subplots(2, 1, figsize=(12, 4*2))
    plot_acf(x_seg, lags=12, title='raw_acf', ax=axes[0])  
    plot_pacf(x_seg, lags=12, title='raw_pacf', ax=axes[1])  
    plt.show()


def find_pq_PQ(ts, m, d, D, max_p=3, max_q=3, max_P=3, max_Q=3):
    best_p, best_q = 0, 0
    best_P, best_Q = 0, 0
    best_bic = np.inf
    best_model = None

    for p in range(max_p):
        for q in range(max_q):
            for P in range(max_P):
                for Q in range(max_Q):
                    print(f"Trying SARIMA({p}, {d}, {q}) x ({P}, {D}, {Q}, {m})")
                    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit(disp=-1)
                    bic = model.bic

                    if bic < best_bic:
                        best_bic = bic
                        best_p = p
                        best_q = q
                        best_P = P
                        best_Q = Q
                        best_model = model

    return best_model, best_p, best_q, best_P, best_Q, best_bic

def arima_fit(x_seq, true, trav=False, model_name="m.pkl"):
    x_diff, d, init = find_d(x_seq, True)
    print(f"diff {d} times")
    # vis_p_acf(x_diff)
    if trav:
        model, bp, bq, bP, bQ, _ = find_pq_PQ(x_diff, 24, d, 2)
        print(f"Best SARIMA({bp}, {d}, {bq}) x ({bP}, {2}, {bQ}, {24})")
    else:
        model = SARIMAX(x_diff, order=(0, d, 1), seasonal_order=(0, 2, 2, 24)).fit(disp=-1)
    # use model to predict
    pred = model.forecast(steps=true.shape[0])
    save_model(model, model_name)
    x_diff = np.concatenate((x_diff, pred))
    y_seq = reverse_diff(x_diff, d, init)
    plt.figure(figsize=(12, 4))
    print(x_seq, true)
    plt.plot(np.concatenate((x_seq, true)), label='original')
    plt.plot(y_seq, label='predicted')
    plt.legend()
    plt.show()

    return y_seq[-true.shape[0]:], model.aic, model.bic


root_path = os.path.dirname("./datasets/")
set_name  = 'ETTh1'
data_path = set_name + '.csv'
dim_use = 7
train_start = 0
train_end = 512
test_span = 64
dim = 6

df_raw = pd.read_csv(os.path.join(root_path, data_path))

for i in range(1, df_raw.shape[1]):
    x = df_raw.iloc[train_start:train_end, i].values
    t = df_raw.iloc[train_end:train_end + test_span, i].values
    arima_fit(x, t, model_name=f"m_{set_name}_{i}.pkl")
    print("dimension ", i)
