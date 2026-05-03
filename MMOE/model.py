# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class CrossLayerLinear(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayerLinear, self).__init__()
        self.linear= nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x

class DCN(nn.Module):
    def __init__(self, input_dim, cross_layers, hidden_units, dropout_rate=0.2):
        super(DCN, self).__init__()
        
        # 交叉网络部分
        self.cross_layers = nn.ModuleList([
            CrossLayerLinear(input_dim) for _ in range(cross_layers)
        ])
        
        # 深度网络部分
        self.deep_layers = nn.ModuleList()
        input_dim_deep = input_dim
        for hidden_unit in hidden_units:
            self.deep_layers.append(nn.Linear(input_dim_deep, hidden_unit))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(dropout_rate))
            input_dim_deep = hidden_unit
        
        # 组合层
        self.combined_layer = nn.Linear(input_dim + hidden_units[-1], 1)

    def forward(self, x):
        # 保存原始输入用于交叉网络
        x0 = x
        
        # 交叉网络
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        
        # 深度网络
        x_deep = x
        for layer in self.deep_layers:
            x_deep = layer(x_deep)
        
        # 拼接输出
        concat_output = torch.cat([x_cross, x_deep], dim=1)
        final_output = self.combined_layer(concat_output)
        
        return final_output


class Expert(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_units_of_expert, expert_output_dim):
        super(Expert, self).__init__()
        self.input_dim = input_dim
        self.experts = nn.ModuleList()
        self.num_experts = num_experts
        for _ in range(self.num_experts):
            layers = []
            dim = self.input_dim 
            for hidden_dim in hidden_units_of_expert:
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                dim = hidden_dim
            # 专家的最后一层不带激活函数  #32
            layers.append(nn.Linear(dim, expert_output_dim))
            self.experts.append(nn.Sequential(*layers))
    def forward(self, shared_input):
        expert_outputs = [expert(shared_input) for expert in self.experts]
        expert_outputs_stack = torch.stack(expert_outputs, dim=1)

        return expert_outputs_stack
    
class Gate(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks):
        super(Gate, self).__init__()
        layer = []
        layer.append(nn.Linear(input_dim, 256))
        layer.append(nn.Linear(256, num_experts))
        self.sequentials = nn.Sequential(*layer)
        self.gates = nn.ModuleList([
            self.sequentials for _ in range(num_tasks)
        ])

    def forward(self, shared_input):
        gate_outputs = [gate(shared_input) for gate in self.gates]
        gate_outputs_stack = torch.stack(gate_outputs, dim=1)
        gate_weights = F.softmax(gate_outputs_stack, dim=2)
        return gate_weights

class Tower(nn.Module):
    def __init__(self, num_tasks, hidden_units_of_tower):
        super(Tower, self).__init__()
        self.num_tasks = num_tasks
        self.towers = nn.ModuleList()
        for _ in range(self.num_tasks):
            # layers = []
            # dim = EXPERT_OUTPUT_DIM
            # for hidden_dim in hidden_units_of_tower:
            #     layers.append(nn.Linear(dim, hidden_dim))
            #     layers.append(nn.BatchNorm1d(hidden_dim))
            #     layers.append(nn.ReLU())
            #     layers.append(nn.Dropout(0.2))
            #     dim = hidden_dim
            # # 塔的最后一层输出一个logit
            # # The last layer of the tower outputs a single logit.
            # layers.append(nn.Linear(dim, 1))    
            # self.towers.append(nn.Sequential(*layers))
            self.towers.append(DCN(EXPERT_OUTPUT_DIM, 2, hidden_units_of_tower))
    def forward(self, task_inputs):
        # [B, 2, 32]
        task_outputs = []
        for i in range(self.num_tasks):
            task_input = task_inputs[:, i, :]          #[B, 1, 32]
            task_output = self.towers[i](task_input)   
            task_outputs.append(task_output)
            
        return task_outputs    #[2]

        

class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) 模型
    此版本为每个网络（专家、门控、塔）定义了独立的结构。
    This version defines separate architectures for each network type (expert, gate, tower).
    """
    def __init__(self, sparse_feature_info, dense_feature_count, emb_dim, hidden_units,
                 tag_vocab_size, num_experts, num_tasks):
        """
        初始化MMoE模型。
        Initialize the MMoE model.

        Args:
            sparse_feature_info (dict): 包含稀疏特征及其词汇表大小的字典。(Dictionary containing sparse features and their vocabulary sizes.)
            dense_feature_count (int): 稠密特征的数量。(Number of dense features.)
            emb_dim (int): 基础Embedding维度。(Base embedding dimension.)
            hidden_units (list): 塔网络和专家网络的隐藏层单元数。(Number of hidden units for tower and expert networks.)
            tag_vocab_size (int): Tag特征的词汇表大小。(Vocabulary size for the tag feature.)
            max_tags (int): Tag序列的最大长度。(Maximum length of the tag sequence.)
            num_experts (int): 专家的数量。(Number of experts.)
            num_tasks (int): 任务的数量。(Number of tasks.)
        """
        super(MMoE, self).__init__()
        self.sparse_feature_info = sparse_feature_info
        self.dense_feature_count = dense_feature_count
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # --- 1. Embedding层 (Embedding Layers) ---
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
            for name, vocab_size in sparse_feature_info.items()
        })
        self.tag_embedding = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=emb_dim)

        # --- 2. 计算总输入维度 (Calculate Total Input Dimension) ---
        sparse_embs_dim = len(self.embedding_layers) * emb_dim
        tag_emb_dim = emb_dim
        self.input_dim = sparse_embs_dim + tag_emb_dim + self.dense_feature_count
        print('输入维度:', self.input_dim)

        # --- 3. 定义专家网络 (Define Expert Networks) ---
        # 专家网络的输入是拼接后的总特征，输出是隐藏层最后一层的维度
        self.experts = Expert(self.input_dim, self.num_experts, hidden_units['expert'], EXPERT_OUTPUT_DIM)

        # --- 4. 定义门控网络 (Define Gate Networks) ---
        # 门控网络的输入是总特征，输出是每个专家的权重
        self.gates = Gate(self.input_dim, self.num_experts, self.num_tasks)

        # --- 5. 定义塔网络 (Define Tower Networks) ---
        # 塔网络的输入是加权组合后的专家输出，输出是该任务的最终预测值
        self.towers = Tower(self.num_tasks, hidden_units['tower'])

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
        
        shared_input = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)

        # --- b. 通过专家网络 (Pass through Experts) ---
        expert_outputs_stack = self.experts(shared_input)

        # --- c. 通过门控网络，计算权重 (Pass through Gates to calculate weights) ---
        gate_weights = self.gates(shared_input)

        # expert_outputs_stack [B, 4, 32]
        # gate_weights         [B, 2, 4]
        # [B, 2, 4]*[B, 4, 32] = [B, 2, 32]
        # --- d. 加权组合并送入塔网络 (Weighted combination and feed into Towers) ---
        task_inputs = torch.bmm(gate_weights, expert_outputs_stack)
        task_outputs = self.towers(task_inputs)
            
        return task_outputs
