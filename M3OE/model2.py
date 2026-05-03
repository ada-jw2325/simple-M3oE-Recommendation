# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from config import * 

class DomainRepresentationLayer(nn.Module):
    """
    实现了M3oE论文中描述的领域表示提取层。
    """
    def __init__(self, input_dim, hidden_dim, num_domains):
        super(DomainRepresentationLayer, self).__init__()
        self.num_domains = num_domains
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 共享参数 (Shared parameters)
        self.W_sh = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_sh = nn.Parameter(torch.empty(hidden_dim))
        
        # 领域专属参数，使用Embedding层高效存储和查找 
        #  (用MAP也行, 但是这样更高效, 不需要列表或者字典去查)
        #  相当于一个查找 领域矩阵的 表
        self.W_d_embedding = nn.Embedding(num_domains, input_dim * hidden_dim)
        self.b_d_embedding = nn.Embedding(num_domains, hidden_dim)
        
        # 初始化参数 (Initialize parameters)
        nn.init.xavier_uniform_(self.W_sh)
        nn.init.zeros_(self.b_sh)
        nn.init.xavier_uniform_(self.W_d_embedding.weight)
        nn.init.zeros_(self.b_d_embedding.weight)

    def forward(self, x, domain_id):
        # x shape: (B, input_dim)
        # domain_id shape: (B,)
        
        # 1. 获取领域专属的权重和偏置
        # Get domain-specific weights and biases
        w_d_flat = self.W_d_embedding(domain_id)  # Shape: (B, input_dim * hidden_dim)  因为从维度角度上说是input_dim -> hidden_dim
        b_d = self.b_d_embedding(domain_id)       # Shape: (B, hidden_dim)
        
        # 2. 将展平的权重恢复成矩阵形式
        # Reshape the flattened weights back into matrix form
        w_d = w_d_flat.view(-1, self.input_dim, self.hidden_dim) # Shape: (B, input_dim, hidden_dim)
        

        # 3. 计算 W_bar_d = W_d * W_sh (利用广播机制)
        # Calculate W_bar_d = W_d * W_sh (using broadcasting)
        # W_sh shape: (input_dim, hidden_dim) -> unsqueezed to (1, input_dim, hidden_dim)
        w_bar_d = w_d * self.W_sh.unsqueeze(0) # Shape: (B, input_dim, hidden_dim)
        
        # 4. 执行 f(x) = W_bar_d * x
        # Perform f(x) = W_bar_d * x using batch matrix multiplication
        # x shape: (B, input_dim) -> unsqueezed to (B, 1, input_dim)
        # (B, 1, input_dim) @ (B, input_dim, hidden_dim) -> (B, 1, hidden_dim)
        h_d = torch.bmm(x.unsqueeze(1), w_bar_d).squeeze(1) # Shape: (B, hidden_dim)
        
        # 5. 添加偏置项
        # Add biases
        h_d = h_d + b_d + self.b_sh
        
        return F.relu(h_d) # 根据论文，通常会有一个激活函数

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class SharedExperts(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_units, output_dim, num_tasks, num_domains):
        super(SharedExperts, self).__init__()
        self.num_tasks = num_tasks
        self.num_domains = num_domains
        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_units, output_dim) for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks * num_domains)
        ])

    def forward(self, x, domain_id, task_id):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_index = domain_id * self.num_tasks + task_id
        gate_weights = torch.zeros(x.size(0), len(self.experts)).to(x.device)
        for i in range(x.size(0)):
            gate_weights[i] = self.gates[gate_index[i]](x[i])
        gate_weights = F.softmax(gate_weights, dim=1).unsqueeze(-1)
        weighted_output = expert_outputs * gate_weights
        return torch.sum(weighted_output, dim=1)

class DomainExperts(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, num_domains):
        super(DomainExperts, self).__init__()
        self.num_domains = num_domains
        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_units, output_dim) for _ in range(num_domains)
        ])
        self.beta_d_params = nn.Parameter(torch.randn(num_domains))

    def forward(self, x, domain_id):
        expert_outputs = [expert(x) for expert in self.experts]
        beta_d = torch.sigmoid(self.beta_d_params)
        output = torch.zeros_like(expert_outputs[0])
        for i in range(x.size(0)):
            d_id = domain_id[i].item()
            current_domain_expert_output = expert_outputs[d_id][i]
            other_domain_expert_outputs = [expert_outputs[j][i] for j in range(self.num_domains) if j != d_id]
            other_sum = torch.stack(other_domain_expert_outputs).mean(dim=0) if other_domain_expert_outputs else 0
            output[i] = beta_d[d_id] * current_domain_expert_output + (1 - beta_d[d_id]) * other_sum
        return output

class TaskExperts(nn.Module):
    # ... (此模块保持不变) ...
    def __init__(self, input_dim, hidden_units, output_dim, num_tasks):
        super(TaskExperts, self).__init__()
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([
            MLP(input_dim, hidden_units, output_dim) for _ in range(num_tasks)
        ])
        self.beta_t_params = nn.Parameter(torch.randn(num_tasks))

    def forward(self, x, task_id):
        expert_outputs = [expert(x) for expert in self.experts]
        beta_t = torch.sigmoid(self.beta_t_params)
        output = torch.zeros_like(expert_outputs[0])
        for i in range(x.size(0)):
            t_id = task_id[i].item()
            current_task_expert_output = expert_outputs[t_id][i]
            other_task_expert_outputs = [expert_outputs[j][i] for j in range(self.num_tasks) if j != t_id]
            other_sum = torch.stack(other_task_expert_outputs).mean(dim=0) if other_task_expert_outputs else 0
            output[i] = beta_t[t_id] * current_task_expert_output + (1 - beta_t[t_id]) * other_sum
        return output

class M3oE(nn.Module):
    """
    Multi-Domain Multi-Task Mixture-of-Experts (M3oE) 模型
    此版本集成了论文中描述的领域表示提取层。
    This version integrates the Domain Representation Extraction Layer described in the paper.
    """
    def __init__(self, sparse_feature_info: Dict, dense_feature_count: int, emb_dim: int, 
                 rep_hidden_dim: int, # 新增: 表示层的隐藏维度 (New: Hidden dimension for the representation layer)
                 expert_hidden_units: List[int], tower_hidden_units: List[int],
                 tag_vocab_size: int, max_tags: int, num_tasks: int, num_domains: int,
                 num_shared_experts: int, domain_feature_name: str):
        super(M3oE, self).__init__()
        
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

        # --- 2. 领域表示提取层 (新) ---
        self.domain_rep_layer = DomainRepresentationLayer(initial_input_dim, rep_hidden_dim, num_domains)

        # --- 3. 专家模块 ---
        # 专家模块的输入维度现在是表示层的输出维度
        # The input dimension for expert modules is now the output dimension of the representation layer.
        expert_input_dim = rep_hidden_dim
        expert_output_dim = expert_hidden_units[-1]
        self.shared_expert_module = SharedExperts(expert_input_dim, num_shared_experts, expert_hidden_units, expert_output_dim, num_tasks, num_domains)
        self.domain_expert_module = DomainExperts(expert_input_dim, expert_hidden_units, expert_output_dim, num_domains)
        self.task_expert_module = TaskExperts(expert_input_dim, expert_hidden_units, expert_output_dim, num_tasks)

        # --- 4. 最终融合权重 (可学习) ---
        self.alpha_d_params = nn.Parameter(torch.randn(num_domains))
        self.alpha_t_params = nn.Parameter(torch.randn(num_tasks))

        # --- 5. 塔网络 ---
        self.towers = nn.ModuleList([
            MLP(expert_output_dim, tower_hidden_units, 1) for _ in range(num_tasks * num_domains)
        ])

    def forward(self, sparse_inputs, dense_inputs, tag_inputs):
        # --- a. 特征处理与拼接 (得到 x_d) ---
        sparse_embs = [
            self.embedding_layers[name](sparse_inputs[:, i])
            for i, name in enumerate(self.sparse_feature_names)
        ]
        sparse_embs_cat = torch.cat(sparse_embs, dim=1)
        
        tag_embs = self.tag_embedding(tag_inputs)
        mask = (tag_inputs != 0).float().unsqueeze(-1)
        pooled_tag_emb = (tag_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        x_d = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)
        
        # --- b. 获取领域ID ---
        domain_id = sparse_inputs[:, self.domain_feature_idx]

        # --- c. 通过领域表示提取层 (得到 h_d) ---
        h_d = self.domain_rep_layer(x_d, domain_id)
        
        # --- d. 通过三个专家模块 ---
        task_outputs = []
        num_tasks = len(self.towers) // self.alpha_d_params.size(0)

        for t_id in range(num_tasks):
            task_id = torch.full_like(domain_id, t_id)
            
            s_output = self.shared_expert_module(h_d, domain_id, task_id)
            d_output = self.domain_expert_module(h_d, domain_id)
            t_output = self.task_expert_module(h_d, task_id)

            # --- e. 最终融合 ---
            alpha_d = torch.sigmoid(self.alpha_d_params)[domain_id].unsqueeze(1)
            alpha_t = torch.sigmoid(self.alpha_t_params)[task_id].unsqueeze(1)
            
            final_representation = s_output + alpha_d * d_output + alpha_t * t_output

            # --- f. 通过对应的塔网络 ---
            tower_logits = torch.zeros(h_d.size(0), 1).to(h_d.device)
            for i in range(h_d.size(0)):
                d_id_val = domain_id[i].item()
                tower_index = d_id_val * num_tasks + t_id
                tower_logits[i] = self.towers[tower_index](final_representation[i].unsqueeze(0))
            
            task_outputs.append(tower_logits)
            
        return task_outputs
