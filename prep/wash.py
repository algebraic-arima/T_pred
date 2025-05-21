import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

root_path = os.path.dirname("./datasets/")
store_path = os.path.dirname("./prep/interp/")
set_name  = 'ETTm1'
data_path = set_name + '.csv'
train_start = 0
train_end = 1000

df_raw = pd.read_csv(os.path.join(root_path, data_path))

x = df_raw.values
origin = x.copy()

# plot the frequency of the data histogram
def plot_frequency(data):
    plt.hist(data, bins=100)
    plt.grid()

def plot_time_series(data):
    for i in range(1, data.shape[1]):
        plt.plot(data[:, i])
    plt.grid()

def vis(data):
    fig, ax = plt.subplots(4, 1, figsize=(16, 12))
    plt.subplot(3, 1, 1)
    plot_time_series(data)
    plt.subplot(3, 1, 2)
    mask, mapp = wash(data)
    print(mapp)
    plt.plot(mask, label='mask')
    plt.legend()
    plt.subplot(3, 1, 3)
    data = data[mask]
    plot_time_series(data)
    plt.show()
    return mask, mapp

def detect_eq(data):
    return (np.diff(data) != 0)

def detect_zero(data, threshold=1.0):
    return (data != 0) | (np.abs(np.roll(data, 1)) < threshold) | (np.abs(np.roll(data, -1)) < threshold)

def wash(data):
    # this function ensures that every segment to be fed / interpolated has length > 1
    mask = np.concatenate(([True], detect_eq(data)))
    mask &= detect_zero(data)
    mapp = {}
    i = 0
    while i < len(data):
        if ~mask[i]:
            start = i
            while i < len(data) and ~mask[i]:
                i += 1
            paddings = max(0, i - start - 20) // 2
            for j in range(start-paddings, i+paddings):
                if j < 0 or j >= len(data):
                    continue
                mapp[j] = j
        else:
            i += 1
    return mask, mapp

def interp(data, start, stride, end=train_end):
    for dim in range(1, data.shape[1]):
        print(f"dim {dim} start")
        x = data[:, dim].copy()
        end = x.shape[0]
        p = np.zeros(end + stride, dtype=float)
        model = SARIMAX(x[:start].astype(np.float64), order=(0, 0, 1), seasonal_order=(0, 1, 2, 24))
        res = model.fit(disp=-1)
        print(f"model {dim} done")
        p[:start] = x[:start]
        k = start
        while k < end:
            print(f"dim {dim} k {k}")
            p[k:k+stride] = res.forecast(steps=stride)
            model = SARIMAX(x[:k + stride].astype(np.float64), order=(0, 0, 1), seasonal_order=(0, 1, 2, 24))
            res = model.fit(disp=-1)
            k += stride
        p = p[:end]
        data[:, dim] = p
        print(f"dim {dim} done")
        df_out = pd.DataFrame(p, columns=[df_raw.columns[dim]])
        df_out.to_csv(os.path.join(store_path, f"{set_name}_{dim}.csv"), index=False)
        plt.figure(figsize=(20, 4))
        plt.plot(origin[:, dim], label='original')
        plt.plot(p, label='interpolated')
        plt.legend()
        plt.show()
        print("MAE", np.mean(np.abs(p - origin[:, dim])))
        print("MSE", np.mean((p - origin[:, dim])**2))
    return data

def predict(data, period):
    for dim in range(6, data.shape[1]):
        print(f"dim {dim} start")
        x = pd.Series(data[:, dim])
        mask, mapp = wash(data[:, dim])
        li = []
        for i, i in mapp.items():
            li.append(i)
            print(f"dim {dim} i {i}")
            llim = max(0, i - period*3)
            history = x[llim:i].astype(np.float64)
            
            if len(history) < period:
                print(f"dim {dim} history too short")
                x[i] = history.mean() if len(history) > 0 else 0
                continue
            try:
                model = ExponentialSmoothing(
                    history,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=period
                )
                fit = model.fit(optimized=True)
                forecast = fit.forecast(1)
                prr = forecast.iloc[0]
                x[i] = prr
            except Exception as e:
                x[i] = history.mean()
                print(f"dim {dim} error: {e} (type: {type(e)}) using history mean")

        # li.reverse()
        # for i in li:
        #     print(f"dim {dim} i {i}")
        #     rlim = min(len(x), i + period*3)
        #     history = x[i+1:rlim][::-1].astype(np.float64)

        #     if len(history) < period:
        #         print(f"dim {dim} history too short")
        #         x[i] = (x[i] + history.mean()) / 2 if len(history) > 0 else 0
        #         continue
        #     try:
        #         model = ExponentialSmoothing(
        #             history,
        #             trend='add',
        #             seasonal='add',
        #             seasonal_periods=24
        #         )
        #         fit = model.fit(optimized=True)
        #         forecast = fit.forecast(1)
        #         x[i] = (x[i] + forecast.iloc[0]) / 2
        #     except Exception as e:
        #         x[i] = (x[i] + history.mean()) / 2
        #         print(f"dim {dim} error: {e} (type: {type(e)}) using history mean")

        data[:, dim] = x
        print(f"dim {dim} done")
        df_out = pd.DataFrame(x, columns=[df_raw.columns[dim]])
        df_out.to_csv(os.path.join(store_path, f"{set_name}_{dim}.csv"), index=False)
        plt.figure(figsize=(20, 4))
        plt.plot(origin[:, dim], label='original')
        plt.plot(x, label='interpolated')
        plt.legend()
        plt.show()

    return data

# mapp: (k, v) means [k, k + v) should be interpolated, and the corresponding mask is False

inte = predict(x, 100)
df_out = pd.DataFrame(inte, columns=df_raw.columns)
df_out.to_csv(os.path.join(store_path, f"{set_name}.csv"), index=False)
