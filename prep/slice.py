import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


root_path = os.path.dirname("./datasets/")
store_path = os.path.dirname("./prep/slice/")
set_name  = 'ETTh1'
data_path = set_name + '.csv'

df_raw = pd.read_csv(os.path.join(root_path, data_path))
x = df_raw.values

l = df_raw.shape[0]

for i in range(l//1000):
    data = x[i*1000:(i+1)*1000, :]
    df = pd.DataFrame(data, columns=df_raw.columns)
    df.to_csv(os.path.join(store_path, f"{set_name}_{i}.csv"), index=False)
