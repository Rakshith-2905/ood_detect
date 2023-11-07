
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import tensordataset
from torch.utils.data import TensorDataset

def get_data_from_saved_files(pickle_file,batch_size=256,train_shuffle=True):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data['train_logits'], list):
        data['train_logits'] = np.concatenate(data['train_logits'])
    if isinstance(data['val_logits'], list):
        data['val_logits'] = np.concatenate(data['val_logits'])
    # data has train_feats, train_logits, train_targets and similarly for val
    #create a Tensor Datasetr

    train_dataset = TensorDataset(torch.from_numpy(data['train_logits']),torch.from_numpy(data['train_features']), torch.from_numpy(data['train_targets']))
    val_dataset = TensorDataset(torch.from_numpy(data['val_logits']),torch.from_numpy(data['val_features']), torch.from_numpy(data['val_targets']))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
