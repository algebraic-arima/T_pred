import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

slope = {
    'ETTm1': {
        1: [
            [15000, 15450, 15],
            [20530, 20677, 212]
        ],
        2:[
            [40715, 41029, 6.5]
        ],
        3: [
            [20527, 20677, -74],
            [55540, 55717, 48]
        ],
        4: [
            [15099, 15524, 10],
            [40682, 41029, 12]
        ],
        5: [
            [15069, 15449, -5.9],
            [20512,20530,12],
            [37148, 38071, 3],
        ],
        7: [
            [26191, 26341, 17.2]
        ]
    }
}

lift = {
    'ETTm1': {
        2: [
            [40715, 41029, 4]
        ],
        6: [
            [10461, 11085, 1.6],
            [11442, 13186, 1.6],
            [32761, 36325, 1.5],
            [43458, 44227, 1.8],
        ]
    }
}

root_path = os.path.dirname("./prep/interp/")
set_name  = 'ETTh1'
origin_path = os.path.dirname("./datasets/")
data_path = set_name + '.csv'

df_origin = pd.read_csv(os.path.join(origin_path, data_path))
x = df_origin.values

for i in range(1, 8):
    n = pd.read_csv(os.path.join(root_path, set_name+ f"_{i}.csv")).values.reshape(-1)
    fig = plt.figure(figsize=(20, 8))
    # compare the n and the df_origin[i]
    plt.plot(x[:,i], label='origin')
    plt.plot(n, label='interp')
    if slope.get(set_name) is not None:
        sl = slope[set_name]
        if sl.get(i) is not None:
            l = sl[i]
            for j in range(len(l)):
                start, end, s = l[j]
                # add a linear, to increase the end by s
                end += 1
                if s >= 0:
                    n[start:end] = n[start:end] + np.linspace(0, s, end-start)
                else:
                    n[start:end] = n[start:end] - np.linspace(0, -s, end-start)
    if lift.get(set_name) is not None:
        li = lift[set_name]
        if li.get(i) is not None:
            l = li[i]
            for j in range(len(l)):
                start, end, s = l[j]
                # lift evenly in the range by s
                end += 1
                n[start:end] = n[start:end] + s
    if slope.get(set_name) is not None or lift.get(set_name) is not None:
        plt.plot(n, label='manual')
    plt.legend()
    plt.show()
    x[:, i] = n


# write the x to a csv file
df = pd.DataFrame(x, columns=df_origin.columns)

df.to_csv(os.path.join(root_path, set_name + '.csv'), index=False)

