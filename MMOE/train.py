import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# 从项目文件中导入必要的模块和配置
from config import *
from datasets import RecDataset
from model import MMoE

def evaluate_mmoe_model(model, dataloader, device):
    """
    在训练过程中评估MMoE模型。
    此版本适配了模型输出logits的最佳实践，并在计算AUC前应用sigmoid。
    同时，它只在点击样本上计算CVR AUC。
    """
    model.eval()  # 设置模型为评估模式
    all_labels = {task: [] for task in TASKS}
    all_preds = {task: [] for task in TASKS}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            tag_inputs = batch['tag_inputs'].to(device)
            labels = batch['labels']
            
            # 模型现在输出的是原始logits
            task_logits = model(sparse_inputs, dense_inputs, tag_inputs)
            
            # 在计算AUC前，需要将logits通过sigmoid转换为概率
            task_preds = [torch.sigmoid(logits) for logits in task_logits]

            for i, task_name in enumerate(TASKS):
                all_labels[task_name].extend(labels[task_name].squeeze().tolist())
                all_preds[task_name].extend(task_preds[i].squeeze().cpu().tolist())

    # --- 计算AUC ---
    auc_scores = {}
    
    # 假设 TASKS = ['click', 'long_view']
    ctr_task_name = TASKS[0]
    cvr_task_name = TASKS[1]
    
    # 1. 计算CTR AUC (在全部样本上)
    auc_scores[f"{ctr_task_name}_auc"] = roc_auc_score(all_labels[ctr_task_name], all_preds[ctr_task_name])

    # 2. 计算CVR AUC (仅在点击样本上)
    all_ctr_labels_np = np.array(all_labels[ctr_task_name])
    clicked_indices = np.where(all_ctr_labels_np == 1)[0]
    
    if len(clicked_indices) > 1 and np.unique(np.array(all_labels[cvr_task_name])[clicked_indices]).size > 1:
        # 筛选出点击样本的CVR标签和预测值
        cvr_labels_clicked = np.array(all_labels[cvr_task_name])[clicked_indices]
        cvr_preds_clicked = np.array(all_preds[cvr_task_name])[clicked_indices]
        
        cvr_auc = roc_auc_score(cvr_labels_clicked, cvr_preds_clicked)
        auc_scores[f"{cvr_task_name}_auc_post_click"] = cvr_auc
    else:
        # 如果点击样本不足或标签全为0/1，无法计算AUC
        auc_scores[f"{cvr_task_name}_auc_post_click"] = 0.5 

    return auc_scores

def main():
    """
    主训练函数
    """
    # --- 1. 加载配置和数据 ---
    print("--- 🚀 开始MMoE模型训练流程 ---")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 确保模型保存目录存在
    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)

    with open(FEATURE_INFO_PATH, 'r') as f:
        feature_max_idx = json.load(f)
    # 仅保留当前模型实际使用的稀疏特征，并按 SPARSE_FEATURES 顺序重排
    feature_max_idx = {name: feature_max_idx[name] for name in SPARSE_FEATURES if name in feature_max_idx}
    
    TAG_VOCAB_SIZE = COUNT_TAGS
    sparse_features = SPARSE_FEATURES
    dense_features = DENSE_FEATURES
    print(f"检测到 {len(sparse_features)} 个稀疏特征和 {len(dense_features)} 个稠密特征。")
    print(f"训练任务: {TASKS}")

    # 创建数据加载器
    train_dataset = RecDataset(TRAIN_PKL_PATH, sparse_features, dense_features)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    test_dataset = RecDataset(TEST_PKL_PATH, sparse_features, dense_features)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- 2. 初始化模型、优化器、损失函数 ---
    print("正在初始化MMoE模型...")
    model = MMoE(
        sparse_feature_info=feature_max_idx,
        dense_feature_count=len(dense_features),
        emb_dim=EMBEDDING_DIM,
        hidden_units=HIDDEN_UNITS,
        tag_vocab_size=TAG_VOCAB_SIZE,
        num_experts=NUM_EXPERTS,
        num_tasks=NUM_TASKS
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 使用BCEWithLogitsLoss以获得更好的数值稳定性，模型应输出原始logits
    loss_fn = nn.BCEWithLogitsLoss() 
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
            
    # --- 3. 训练与评估循环 ---
    print("开始训练...")
    best_avg_auc = 0.0
    for epoch in range(EPOCHS):
        model.train() # 设置模型为训练模式
        total_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch in train_iterator:
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            tag_inputs = batch['tag_inputs'].to(device)
            labels = batch['labels']

            optimizer.zero_grad()
            # 模型输出的是原始logits
            task_logits = model(sparse_inputs, dense_inputs, tag_inputs)
            
            # --- 修正后的损失计算逻辑 ---
            ctr_label = labels[TASKS[0]].to(device)
            cvr_label = labels[TASKS[1]].to(device)
            
            # 1. CTR loss (在全体曝光样本上计算)
            ctr_loss = loss_fn(task_logits[0], ctr_label)

            # 2. CVR loss (只在点击样本上计算，解决SSB问题)
            click_mask = (ctr_label == 1).squeeze(1)
            
            if click_mask.sum() > 0:
                cvr_loss = loss_fn(
                    task_logits[1][click_mask], # 只选择被点击样本的CVR logits
                    cvr_label[click_mask]       # 只选择被点击样本的CVR真实标签
                )
                current_loss = ctr_loss + cvr_loss # 这里可以考虑给cvr_loss加权重, e.g., 0.5 * cvr_loss
            else:
                # 如果当前batch没有点击样本，则只用ctr_loss进行反向传播
                current_loss = ctr_loss
            
            current_loss.backward()
            optimizer.step()
            total_loss += current_loss.item()
            
            train_iterator.set_postfix({
                "loss": f"{total_loss / (train_iterator.n + 1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # --- 在每个epoch结束后，进行评估 ---
        test_auc_scores = evaluate_mmoe_model(model, test_loader, device)
        avg_auc = np.mean(list(test_auc_scores.values()))
        
        print(f"Epoch {epoch+1} 结束. 平均损失: {total_loss / len(train_loader):.4f} - 测试集平均AUC: {avg_auc:.4f}")
        for task, auc in test_auc_scores.items():
            if "post_click" in task:
                print(f"  - CVR_AUC (点击后): {auc:.4f}")
            else:
                print(f"  - {task.upper()}: {auc:.4f}")

        # --- 保存最优模型 ---
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_PATH, "mmoe_best_model.pth"))
            print(f"🎉 新的最优模型已保存! 平均AUC: {best_avg_auc:.4f}")

        # 更新学习率
        scheduler.step()

    print(f"\n✅ 训练完成！最优模型已保存至: {SAVED_MODEL_PATH} (最优平均AUC: {best_avg_auc:.4f})")

if __name__ == '__main__':
    main()