# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class CGC_Layer(nn.Module):
    """
    一个独立的PLE抽取网络层.
    它包含一组共享专家和多组任务专属专家，以及对应的门控网络。
    """
    def __init__(self, input_dim: int, num_tasks: int, num_shared_experts: int, 
                 num_task_experts: int, expert_hidden_units: List[int]):
        super(CGC_Layer, self).__init__()
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts  #[List]

        # --- 定义专家网络 (Define Expert Networks) ---
        expert_output_dim = expert_hidden_units[-1]
        
        # 共享专家 (Shared Experts)
        self.shared_experts = nn.ModuleList([
            self.create_expert_network(input_dim, expert_hidden_units, expert_output_dim)
            for _ in range(self.num_shared_experts)  # 8
        ])
        
        # 任务专属专家 (Task-specific Experts)
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                self.create_expert_network(input_dim, expert_hidden_units, expert_output_dim)
                for _ in range(self.num_task_experts) # 4
            ]) for _ in range(self.num_tasks)  # 2 
        ])

        # --- 定义门控网络 (Define Gate Networks) ---
        # 每个任务一个门控，控制其专属专家和共享专家
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, self.num_task_experts + self.num_shared_experts)
            for _ in range(self.num_tasks) # 2
        ])

    def create_expert_network(self, input_dim, hidden_units, output_dim):
        """辅助函数 用于创建一个MLP专家网络"""
        layers = []
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, task_inputs: List[torch.Tensor], shared_input: torch.Tensor):
        """
        CGC层的前向传播。
        Args:
            task_inputs (List[torch.Tensor]): 一个列表，包含每个任务上一层的输出。
            shared_input (torch.Tensor): 共享专家的输入。
        Returns:
            List[torch.Tensor]: 一个列表，包含该层为每个任务生成的输出。
        """
        # --- 通过专家网络 ---
        shared_expert_outputs = [expert(shared_input) for expert in self.shared_experts]
        
        task_expert_outputs = []
        for i in range(self.num_tasks): # 2
            # 刚开始shared 为task一样的
            # 后来是和AB专家的输出 合并的
            task_specific_outputs = [self.task_experts[i][j](task_inputs[i]) for j in range(self.num_task_experts)] # 4
            task_expert_outputs.append(task_specific_outputs)

        # --- 通过门控网络并进行加权组合 ---
        final_task_outputs = []
        for i in range(self.num_tasks):
            # 1. 组合当前任务需要考虑的所有专家输出
            current_experts_outputs = task_expert_outputs[i] + shared_expert_outputs
            current_experts_stack = torch.stack(current_experts_outputs, dim=1) # Shape: (B, N_task + N_shared, D_expert)

            # 2. 计算门控权重
            gate_input = task_inputs[i] # 门控的输入是对应任务的输入 (Gate input is the input for the corresponding task)
            gate_weights = F.softmax(self.gates[i](gate_input), dim=1).unsqueeze(-1) # Shape: (B, N_task + N_shared, 1)
            # 本质和矩阵相乘一样, 那里用的是bmm, 这里用广播相乘是一样的
            # 3. 加权求和
            weighted_output = current_experts_stack * gate_weights # Broadcasting  (B, N_task + N_shared, D_expert)
            task_output = torch.sum(weighted_output, dim=1) # Shape: (B, D_expert)
            final_task_outputs.append(task_output)
            
        return final_task_outputs  # [(B, D_expert), (B, D_expert)]


class PLE(nn.Module):
    """
    Progressive Layered Extraction (PLE) 模型
    """
    def __init__(self, sparse_feature_info: Dict, dense_feature_count: int, emb_dim: int, 
                 cgc_hidden_units,
                 tag_vocab_size: int, num_tasks: int, 
                 num_shared_experts: int, num_task_experts: int):
        super(PLE, self).__init__()
        self.num_tasks = num_tasks

        # --- 1. 底层Embedding (Bottom Embedding Layer) ---
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
            for name, vocab_size in sparse_feature_info.items() if name != 'tag'
        })
        self.tag_embedding = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=emb_dim)
        
        # 计算初始输入维度
        initial_input_dim = len(self.embedding_layers) * emb_dim + emb_dim + dense_feature_count

        # --- 2. 抽取网络层 (Extraction Network Layers) ---
        # 我们将创建两个CGC层
        self.cgc_layers = nn.ModuleList()
        
        # 第一层 (First Layer)
        self.cgc_layers.append(
            CGC_Layer(initial_input_dim, num_tasks, num_shared_experts, num_task_experts, cgc_hidden_units['1'])
        )
        # 第二层 (Second Layer)
        # 第二层的输入维度是第一层专家的输出维度
        second_layer_input_dim = cgc_hidden_units['1'][-1]
        self.cgc_layers.append(
            CGC_Layer(second_layer_input_dim, num_tasks, num_shared_experts, num_task_experts, cgc_hidden_units['2'])
        )

        # --- 3. 塔网络 (Tower Networks) ---
        tower_input_dim = cgc_hidden_units['2'][-1]
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            layers = []
            dim = tower_input_dim
            for hidden_dim in cgc_hidden_units['tower']:
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                dim = hidden_dim
            layers.append(nn.Linear(dim, 1))
            self.towers.append(nn.Sequential(*layers))

    def forward(self, sparse_inputs, dense_inputs, tag_inputs):
        # --- a. 特征处理与拼接 (Feature Processing and Concatenation) ---
        sparse_embs = [
            self.embedding_layers[name](sparse_inputs[:, i])
            for i, name in enumerate(self.embedding_layers.keys())
        ]
        sparse_embs_cat = torch.cat(sparse_embs, dim=1)
        
        tag_embs = self.tag_embedding(tag_inputs)
        mask = (tag_inputs != 0).float().unsqueeze(-1)
        pooled_tag_emb = (tag_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        initial_input = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)

        # --- b. 逐层通过CGC网络 (Pass through CGC layers progressively) ---
        # 第一层的输入
        # Input for the first layer
        task_inputs_l1 = [initial_input] * self.num_tasks
        shared_input_l1 = initial_input
        
        # 通过第一层
        # Pass through the first layer
        task_outputs_l1 = self.cgc_layers[0](task_inputs_l1, shared_input_l1)
        
        # 第二层的输入
        # Input for the second layer
        task_inputs_l2 = task_outputs_l1
        shared_input_l2 = sum(task_outputs_l1) # 将所有任务的输出相加作为共享输入 [B, ]
        # 通过第二层
        # Pass through the second layer
        final_task_outputs = self.cgc_layers[1](task_inputs_l2, shared_input_l2)

        # --- c. 通过塔网络得到最终预测 (Pass through towers for final prediction) ---
        results = []
        for i in range(len(self.towers)):
            output = self.towers[i](final_task_outputs[i])
            output = torch.sigmoid(output)
            results.append(output)
            
        return results
