import argparse
import os
import pandas as pd
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import load_args, string_split

def create_batches(time_series, seq_length=600, step=3):
    n = time_series.shape[0]
    if n <= seq_length:
        raise ValueError(f"must > {seq_length}")
    
    batches = []
    
    for i in range(0, n - seq_length, step):
        batch = time_series[i:i+seq_length]
        batches.append(batch)
    
    return batches

parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--checkpoint_root', type=str, default='./checkpoints', help='location of the trained model')
parser.add_argument('--setting_name', type=str, default='Crossformer_ETTh1_il168_ol24_sl6_win2_fa10_dm256_nh4_el3_itr0', help='name of the experiment')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--different_split', action='store_true', help='use data split different from training process', default=False)
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='data split of train, vali, test')

parser.add_argument('--inverse', action='store_true', help='inverse output data into the original scale', default=False)
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False
args.device = torch.device('cuda:{}'.format(args.gpu) if args.use_gpu else 'cpu')

args.checkpoint_dir = os.path.join(args.checkpoint_root, args.setting_name)
hyper_parameters = load_args(os.path.join(args.checkpoint_dir, 'args.json'))

#load the pre-trained model
args.data_dim = hyper_parameters['data_dim']; args.in_len = hyper_parameters['in_len']; args.out_len = hyper_parameters['out_len'];
args.seg_len = hyper_parameters['seg_len']; args.win_size = hyper_parameters['win_size']; args.factor = hyper_parameters['factor'];
args.d_model = hyper_parameters['d_model']; args.d_ff = hyper_parameters['d_ff']; args.n_heads = hyper_parameters['n_heads'];
args.e_layers = hyper_parameters['e_layers']; args.dropout = hyper_parameters['dropout']; args.baseline = hyper_parameters['baseline'];
exp = Exp_crossformer(args)
model_dict = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location='cpu')
exp.model.load_state_dict(model_dict)

df = pd.read_csv("./datasets/BACH.csv")
x = df.values[:, 1:]
x = x.astype(np.float32)
print(x.shape) # (241063, 8)
b = create_batches(x) # 240463 len, (600, 8)
print(len(b)) # 80155
batch_size = 41 # 1955 batches
pred = []
for i in range(0, len(b), batch_size):
    ba = np.stack(b[i:i+batch_size])
    ba = torch.from_numpy(ba)
    ba = ba.float().to(args.device)
    pred.append(exp.model(ba).detach().cpu().numpy()[:,:,:]) # (41,3,8)
    print(len(pred)) # 1, 2, 3, ..., 1955

pred = np.concatenate(pred, axis=0)
print(pred.shape) # (41*1955, 3, 8)
np.save("./datasets/BACH_pred.npy", pred)
print(x.shape) # (60266, 8)
p = np.load("./datasets/BACH_pred.npy")
print(p.shape) # (41*1955, 3, 8)
p = np.reshape(p, (-1, 8)) # (41*1955*3, 8)
p = np.concatenate((x[:600], p), axis=0) # (41*1955*3+600=60266, 8)
print(p.shape) # (241065, 8)


from scipy.io import wavfile

audio_data = p.copy()
repeat_times = 5
audio_data = np.repeat(audio_data, repeat_times, axis=0) 

sample_rate = 30000

S_L, S_R = audio_data[:, 0], audio_data[:, 1]
A_L, A_R = audio_data[:, 2], audio_data[:, 3]
T_L, T_R = audio_data[:, 4], audio_data[:, 5]
B_L, B_R = audio_data[:, 6], audio_data[:, 7]


def normalize(x):
    return x / np.max(np.abs(x))

S_L, S_R = normalize(S_L), normalize(S_R)
A_L, A_R = normalize(A_L), normalize(A_R)
T_L, T_R = normalize(T_L), normalize(T_R)
B_L, B_R = normalize(B_L), normalize(B_R)


S_stereo = np.column_stack((S_L, S_R))
A_stereo = np.column_stack((A_L, A_R))
T_stereo = np.column_stack((T_L, T_R))
B_stereo = np.column_stack((B_L, B_R))

wavfile.write("S.wav", sample_rate, S_stereo.astype(np.float32))
wavfile.write("A.wav", sample_rate, A_stereo.astype(np.float32))
wavfile.write("T.wav", sample_rate, T_stereo.astype(np.float32))
wavfile.write("B.wav", sample_rate, B_stereo.astype(np.float32))


