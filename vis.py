import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_path = os.path.dirname("./datasets/")
set_name  = 'ETTm1'
data_path = set_name + '.csv'
dim_use = 7
time_span = 1024

fig_path = os.path.join(root_path, 'figures/' + set_name + '/')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

df_raw = pd.read_csv(os.path.join(root_path, data_path))

ind = np.linspace(0, len(df_raw) - 1, len(df_raw), dtype=int)

for start in range(0, len(df_raw), time_span):
    plt.figure(figsize=(20, 8))
    for i in range(1, df_raw.shape[1]):
        plt.plot(ind[start:start + time_span], df_raw.iloc[start:start + time_span, i], label=df_raw.columns[i])
        plt.legend()
        if i == dim_use:
            break
        
    plt.savefig(os.path.join(fig_path, f"{set_name}_{start}.png"))


# dim_i, dim_j = 5, 7
# plt.figure(figsize=(20, 8))
# plt.scatter(df_raw.iloc[time_start:time_start + time_span, dim_i], df_raw.iloc[time_start:time_start + time_span, dim_j])
# plt.savefig(os.path.join(fig_path, f"{set_name}_{dim_i}_{dim_j}.png"))
