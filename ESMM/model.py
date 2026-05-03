# model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    """一个简单的多层感知机模块"""
    def __init__(self, input_dim, hidden_units, dropout_rate=0.1):
        super(MLP, self).__init__()
        layers = []
        for hidden_unit in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_unit
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class MLPP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super(MLPP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers = []

        dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            dim = hidden_dim
        
        layers.append(nn.Linear(dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x): 
        return self.layers(x)



    
class CrossLayerLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x0, x_l):
        # x0: 初始输入, x_l: 上一层输出
        return x0 * self.linear(x_l) + x_l

class DCN(nn.Module):
    def __init__(self, input_dim, cross_layers, hidden_units, dropout_rate=0.2):
        super(DCN, self).__init__()
        
        # 交叉网络部分
        self.cross_layers = nn.ModuleList([
            CrossLayerLinear(input_dim, _) for _ in range(cross_layers)
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
    
class ESMM(nn.Module):
    def __init__(self, sparse_feature_info, dense_feature_count, emb_dim, hidden_units, tag_vocab_size, max_tags=10):
        super(ESMM, self).__init__()
        self.sparse_feature_info = sparse_feature_info
        self.dense_feature_count = dense_feature_count
        
        #TODO: 为不同特征选取不同的嵌入维度, 可以让ID类的维度大一些(比如256), 让普通特征的维度小一些 
        self.emb_dim = emb_dim

        # Embedding层
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
            for name, vocab_size in sparse_feature_info.items()
        })

        # 为tags层做额外的Embedding层
        self.tag_embedding = nn.Embedding(num_embeddings=tag_vocab_size, embedding_dim=emb_dim)


        # 计算MLP的输入维度
        mlp_input_dim = len(sparse_feature_info) * emb_dim + emb_dim + dense_feature_count
        print('input', mlp_input_dim)
        # 定义CTR和CVR两个Tower
        self.cvr_tower = MLP(mlp_input_dim, hidden_units)
        self.ctr_tower = MLP(mlp_input_dim, hidden_units)

    def forward(self, sparse_inputs, dense_inputs, tag_inputs):
        # 获取稀疏特征的embedding
        sparse_embs = [
            self.embedding_layers[name](sparse_inputs[:, i])
            for i, name in enumerate(self.sparse_feature_info.keys())
        ]
        sparse_embs_cat = torch.cat(sparse_embs, dim=1)

        tag_embs = self.tag_embedding(tag_inputs)
        mask = (tag_inputs != 0).float().unsqueeze(-1)
        pooled_tag_emb = (tag_embs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 拼接稠密特征
        shared_input = torch.cat([sparse_embs_cat, pooled_tag_emb, dense_inputs], dim=1)
        # CTR Tower
        ctr_logits = self.ctr_tower(shared_input)
        p_ctr = torch.sigmoid(ctr_logits)

        # CVR Tower
        cvr_logits = self.cvr_tower(shared_input)
        p_cvr = torch.sigmoid(cvr_logits)
        
        # 计算CTCVR
        p_ctcvr = p_ctr * p_cvr # 这是ESMM的核心
        
        return p_ctr, p_ctcvr