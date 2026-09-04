import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

data_path = '../drafts/data'

def read_idx3_images(path):
    with open(path, 'rb') as f:
        # skip the 16 byte header: magic number(4), count(4), rows(4), cols(4)
        data = np.fromfile(f, dtype=np.uint8, offset=16)
    # reshape to (#images, 784) 
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def read_idx1_labels(path):
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8, offset=8)
    return data.astype(np.int64)

def getDataLoader(split, batch_size):
    images = read_idx3_images(data_path + '/train-images-idx3-ubyte') if (split == 'train') else read_idx3_images(data_path + '/t10k-images-idx3-ubyte')
    labels = read_idx1_labels(data_path + '/train-labels-idx1-ubyte') if (split == 'train') else read_idx1_labels(data_path + '/t10k-labels-idx1-ubyte')
    
    x_train = torch.tensor(images, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)
    train_ds = TensorDataset(x_train, y_train)
    
    print(x_train.shape)
    print(f"1 epoch = {10000 / batch_size} batches")
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)