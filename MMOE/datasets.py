# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RecDataset(Dataset):
    def __init__(self, data_path, sparse_features, dense_features):
        super().__init__()
        self.data_df = pd.read_pickle(data_path)

        self.sparse_features = sparse_features
        self.dense_features = dense_features
        # 转换为numpy数组以提高效率
        self.sparse_inputs = self.data_df[self.sparse_features].values
        self.dense_inputs = self.data_df[self.dense_features].values

        self.tag_inputs = np.array(self.data_df['tag_processed'].tolist(), dtype=np.int64)

        self.ctr_labels = self.data_df['ctr_label'].values
        self.cvr_labels = self.data_df['cvr_label'].values
        self.ctcvr_labels = self.data_df['ctcvr_label'].values
        
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # 将Numpy数组转换为PyTorch Tensors
        sparse_vals = torch.LongTensor(self.sparse_inputs[idx])
        dense_vals = torch.FloatTensor(self.dense_inputs[idx])
        
        tag_vals = torch.LongTensor(self.tag_inputs[idx])
        
        ctr_label = torch.FloatTensor([self.ctr_labels[idx]])
        cvr_label = torch.FloatTensor([self.cvr_labels[idx]])
        ctcvr_label = torch.FloatTensor([self.ctcvr_labels[idx]])
        
        return {
            "sparse_inputs": sparse_vals,
            "dense_inputs": dense_vals,
            "tag_inputs": tag_vals,
            "labels":{'ctr':ctr_label, 'cvr':cvr_label}
        }