import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_path = os.path.dirname("./datasets/")
set_name  = 'ECL'
data_path = set_name + '.csv'
dim_use = 7
time_start = 10900
time_span = 32

fig_path = os.path.join(root_path, 'figures')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

df_raw = pd.read_csv(os.path.join(root_path, data_path))

hh_old = None
for i in range(df_raw.shape[0]):
    date, time = df_raw.iloc[i, 0].split(' ')
    y, mo, d = date.split('-')
    h, mi, s = time.split(':')
    hh = int(d) * 24 + int(h)
    if hh_old is not None:
        if int(d) != 1 and hh != hh_old + 1:
            print("time error at "+ date + ' ' + time + ' ' + str(hh) + ' ' + str(hh_old))
            break
        elif int(d) == 1 and h == 0:
            hh_old = 0
    hh_old = hh

plt.figure(figsize=(20, 5))
for i in range(1, 1 + dim_use):
    plt.plot(df_raw.iloc[time_start:time_start + time_span, i], label = df_raw.columns[i])
    plt.legend()
    
plt.savefig(os.path.join(fig_path, set_name + '_raw.png'))