# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from config import * 

class MLP(nn.Module):
    """
    一个通用的MLP模块 特别处理了BatchNorm在batch size为1时的问题。
    """
    def __init__(self, input_dim, hidden_units, output_dim=1, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # ==================== 关键修改 (Key Change) ====================
            # 使用LayerNorm，它不依赖于batch size
            # Use LayerNorm, which is independent of batch size.
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TowerMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1, dropout_rate=0.2):
        super(TowerMLP, self).__init__()
        layers = []
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SharedExperts(nn.Module):
    """共享专家模块 (Shared Expert Module) - 高效向量化版"""
    def __init__(self, input_dim, num_experts, hidden_units, output_dim, num_tasks, num_domains):
        super(SharedExperts, self).__init__()
        self.num_tasks, self.num_domains, self.num_experts = num_tasks, num_domains, num_experts
        self.experts = nn.ModuleList([MLP(input_dim, hidden_units, output_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_tasks * num_domains)]
                                   )
        
    def forward(self, x, domain_id, task_id):
        # 1. 获取所有专家的输出 (Get all expert outputs)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 2. 高效计算门控权重 (Efficiently compute gate weights)
        gate_index = domain_id * self.num_tasks + task_id
        gate_logits = torch.zeros(x.size(0), self.num_experts, device=x.device)
        
        for i in range(len(self.gates)):
            mask = (gate_index == i)
            if not mask.any(): continue
            gate_logits[mask] = self.gates[i](x[mask])
            
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(-1)
        
        # 3. 加权求和 (Weighted sum)
        return torch.sum(expert_outputs * gate_weights, dim=1)

class DomainExperts(nn.Module):
    """领域专家模块 (Domain Expert Module) - 高效向量化版"""
    def __init__(self, input_dim, hidden_units, output_dim, num_domains):
        super(DomainExperts, self).__init__()
        self.num_domains = num_domains
        self.experts = nn.ModuleList([MLP(input_dim, hidden_units, output_dim) for _ in range(num_domains)])
        self.beta_d_params = nn.Parameter(torch.randn(num_domains))

    def forward(self, x, domain_id):
        expert_outputs_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        beta_d = torch.sigmoid(self.beta_d_params)
        output = torch.zeros_like(expert_outputs_stack[:, 0, :])
        
        for d_id in range(self.num_domains):
            mask = (domain_id == d_id)
            if not mask.any(): continue
            
            current_domain_out = expert_outputs_stack[mask, d_id, :]
            other_expert_indices = [j for j in range(self.num_domains) if j != d_id]
            
            if other_expert_indices:
                other_domain_outs = expert_outputs_stack[mask][:, other_expert_indices, :].mean(dim=1)
            else:
                other_domain_outs = 0
            
            weighted_sum = beta_d[d_id] * current_domain_out + (1 - beta_d[d_id]) * other_domain_outs
            output[mask] = weighted_sum
            
        return output

class TaskExperts(nn.Module):
    """任务专家模块 (Task Expert Module) - 高效向量化版"""
    def __init__(self, input_dim, hidden_units, output_dim, num_tasks):
        super(TaskExperts, self).__init__()
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP(input_dim, hidden_units, output_dim) for _ in range(num_tasks)])
        self.beta_t_params = nn.Parameter(torch.randn(num_tasks))

    def forward(self, x, task_id): # task_id is a scalar int
        expert_outputs = [expert(x) for expert in self.experts]
        beta_t = torch.sigmoid(self.beta_t_params)
        
        current_task_out = expert_outputs[task_id]
        
        other_task_indices = [j for j in range(self.num_tasks) if j != task_id]
        if other_task_indices:
            other_task_outs_mean = torch.stack([expert_outputs[j] for j in other_task_indices]).mean(dim=0)
        else:
            other_task_outs_mean = 0
            
        return beta_t[task_id] * current_task_out + (1 - beta_t[task_id]) * other_task_outs_mean

class DomainRepresentationLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains):
        super(DomainRepresentationLayer, self).__init__()
        self.num_domains, self.input_dim, self.hidden_dim = num_domains, input_dim, hidden_dim
        self.W_sh = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_sh = nn.Parameter(torch.empty(hidden_dim))
        self.W_d_embedding = nn.Embedding(num_domains, input_dim * hidden_dim)
        self.b_d_embedding = nn.Embedding(num_domains, hidden_dim)
        nn.init.xavier_uniform_(self.W_sh); nn.init.zeros_(self.b_sh)
        nn.init.xavier_uniform_(self.W_d_embedding.weight); nn.init.zeros_(self.b_d_embedding.weight)

    def forward(self, x, domain_id):
        w_d_flat = self.W_d_embedding(domain_id)
        b_d = self.b_d_embedding(domain_id)
        w_d = w_d_flat.view(-1, self.input_dim, self.hidden_dim)
        w_bar_d = w_d * self.W_sh.unsqueeze(0)
        h_d = torch.bmm(x.unsqueeze(1), w_bar_d).squeeze(1)
        h_d = h_d + b_d + self.b_sh
        return F.relu(h_d)

class M3oE(nn.Module):
    """
    Multi-Domain Multi-Task Mixture-of-Experts (M3oE) 模型
    此版本使用向量化操作以提升性能。
    This version uses vectorized operations for better performance.
    """
    def __init__(self, sparse_feature_info: Dict, dense_feature_count: int, emb_dim: int, 
                 rep_hidden_dim: int, expert_hidden_units: List[int], tower_hidden_units: List[int],
                 tag_vocab_size: int, num_tasks: int, num_domains: int,
                 num_shared_experts: int, domain_feature_name: str):
        super(M3oE, self).__init__()
        
        self.num_tasks, self.num_domains = num_tasks, num_domains
        self.domain_feature_name = domain_feature_name
        self.sparse_feature_names = SPARSE_FEATURES
        self.domain_feature_idx = self.sparse_feature_names.index(domain_feature_name)

        # --- 1. Embedding层 ---
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
            for name, vocab_size in sparse_feature_info.items() if name != 'tag'
        })
        self.tag_embedding = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=emb_dim)
        
        initial_input_dim = len(self.embedding_layers) * emb_dim + emb_dim + dense_feature_count

        # --- 2. 领域表示提取层 ---
        self.domain_rep_layer = DomainRepresentationLayer(initial_input_dim, rep_hidden_dim, num_domains)

        # --- 3. 专家模块 ---
        expert_input_dim = rep_hidden_dim
        expert_output_dim = expert_hidden_units[-1]
        self.shared_expert_module = SharedExperts(expert_input_dim, num_shared_experts, expert_hidden_units, expert_output_dim, num_tasks, num_domains)
        self.domain_expert_module = DomainExperts(expert_input_dim, expert_hidden_units, expert_output_dim, num_domains)
        self.task_expert_module = TaskExperts(expert_input_dim, expert_hidden_units, expert_output_dim, num_tasks)

        # --- 4. 最终融合权重---
        self.alpha_d_params = nn.Parameter(torch.randn(num_domains))
        self.alpha_t_params = nn.Parameter(torch.randn(num_tasks))

        # --- 5. 塔网络 ---
        self.towers = nn.ModuleList([
            TowerMLP(expert_output_dim, tower_hidden_units, 1) for _ in range(num_tasks * num_domains)
        ])

    def forward(self, sparse_inputs, dense_inputs, tag_inputs):
        # --- a. 特征处理与拼接 (得到 x_d) ---
        sparse_embs = [self.embedding_layers[name](sparse_inputs[:, i]) for i, name in enumerate(self.sparse_feature_names)]
        sparse_embs_cat = torch.cat(sparse_embs, dim=1)
        tag_embs = self.tag_embedding(tag_inputs)

        mask = (tag_inputs != 0).float().unsqueeze(-1)
        pooled_tag_emb = (tag_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        x_d = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)
        
        domain_id = sparse_inputs[:, self.domain_feature_idx]
        h_d = self.domain_rep_layer(x_d, domain_id)
        
        # --- b. 循环处理每个任务 ---
        task_outputs = []
        for t_id in range(self.num_tasks):
            task_id_tensor = torch.full_like(domain_id, t_id)
            
            s_output = self.shared_expert_module(h_d, domain_id, task_id_tensor)
            d_output = self.domain_expert_module(h_d, domain_id)
            t_output = self.task_expert_module(h_d, t_id) # 传递标量t_id

            alpha_d = torch.sigmoid(self.alpha_d_params)[domain_id].unsqueeze(1)
            alpha_t = torch.sigmoid(self.alpha_t_params)[t_id]
            
            final_representation = s_output + alpha_d * d_output + alpha_t * t_output

            # --- c. 向量化塔网络计算 ---
            tower_logits = torch.zeros(h_d.size(0), 1, device=h_d.device)
            for d_id_val in range(self.num_domains):
                domain_mask = (domain_id == d_id_val)
                if not domain_mask.any(): continue
                
                tower_index = d_id_val * self.num_tasks + t_id
                inputs_for_this_tower = final_representation[domain_mask]
                outputs_for_this_tower = self.towers[tower_index](inputs_for_this_tower)
                tower_logits[domain_mask] = outputs_for_this_tower
            
            task_outputs.append(tower_logits)
            
        return task_outputs
