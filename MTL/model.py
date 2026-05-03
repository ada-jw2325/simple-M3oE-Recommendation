# model.py
import torch
import torch.nn as nn
from typing import List, Dict

class SharedBottom(nn.Module):
    """
    传统的Shared-Bottom多任务学习模型。
    A traditional Shared-Bottom Multi-Task Learning model.
    """
    def __init__(self, sparse_feature_info: Dict, dense_feature_count: int, emb_dim: int, 
                 bottom_hidden_units: List[int], tower_hidden_units: List[int],
                 tag_vocab_size: int, num_tasks: int):
        """
        初始化Shared-Bottom模型。
        Initialize the Shared-Bottom model.

        Args:
            sparse_feature_info (Dict): 包含稀疏特征及其词汇表大小的字典。
            dense_feature_count (int): 稠密特征的数量。
            emb_dim (int): 基础Embedding维度。
            bottom_hidden_units (List[int]): 共享底层网络的隐藏层单元数。
            tower_hidden_units (List[int]): 任务塔网络的隐藏层单元数。
            tag_vocab_size (int): Tag特征的词汇表大小。
            max_tags (int): Tag序列的最大长度。
            num_tasks (int): 任务的数量。
        """
        super(SharedBottom, self).__init__()

        # --- 1. Embedding层 (Embedding Layers) ---
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
            for name, vocab_size in sparse_feature_info.items() if name != 'tag'
        })
        self.tag_embedding = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=emb_dim)
        
        # 计算初始输入维度
        initial_input_dim = len(self.embedding_layers) * emb_dim + emb_dim + dense_feature_count

        # --- 2. 共享底层网络 (Shared Bottom Network) ---
        bottom_layers = []
        dim = initial_input_dim
        for hidden_dim in bottom_hidden_units:
            bottom_layers.append(nn.Linear(dim, hidden_dim))
            bottom_layers.append(nn.BatchNorm1d(hidden_dim))
            bottom_layers.append(nn.ReLU())
            bottom_layers.append(nn.Dropout(0.2))
            dim = hidden_dim
        self.shared_bottom = nn.Sequential(*bottom_layers)

        # --- 3. 任务专属塔网络 (Task-specific Towers) ---
        tower_input_dim = bottom_hidden_units[-1]
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower_layers = []
            dim = tower_input_dim
            for hidden_dim in tower_hidden_units:
                tower_layers.append(nn.Linear(dim, hidden_dim))
                tower_layers.append(nn.BatchNorm1d(hidden_dim))
                tower_layers.append(nn.ReLU())
                tower_layers.append(nn.Dropout(0.2))
                dim = hidden_dim
            tower_layers.append(nn.Linear(dim, 1))
            self.towers.append(nn.Sequential(*tower_layers))

    def forward(self, sparse_inputs, dense_inputs, tag_inputs):
        # --- a. 特征处理与拼接 ---
        sparse_embs = [
            self.embedding_layers[name](sparse_inputs[:, i])
            for i, name in enumerate(self.embedding_layers.keys())
        ]
        sparse_embs_cat = torch.cat(sparse_embs, dim=1)
        
        tag_embs = self.tag_embedding(tag_inputs)
        mask = (tag_inputs != 0).float().unsqueeze(-1)
        pooled_tag_emb = (tag_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        initial_input = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)

        # --- b. 通过共享底层 ---
        bottom_output = self.shared_bottom(initial_input)
        
        # --- c. 将共享输出分发到各个塔网络 ---
        task_outputs = []
        for tower in self.towers:
            task_output = tower(bottom_output)
            task_output = torch.sigmoid(task_output)
            task_outputs.append(task_output)
            
        return task_outputs
