# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json, os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 从项目文件中导入必要的模块和配置
from config import *
from datasets import RecDataset
from model import M3oE # 导入我们新的M3oE模型

def evaluate_m3oe_model(model, dataloader, device):
    """
    在训练过程中评估M3oE模型。
    Evaluates the M3oE model during training.
    """
    model.eval()
    all_labels = {task: [] for task in TASKS}
    all_preds = {task: [] for task in TASKS}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="正在验证", leave=False):
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            tag_inputs = batch['tag_inputs'].to(device)
            labels = batch['labels']
            
            task_logits = model(sparse_inputs, dense_inputs, tag_inputs)
            task_preds = [torch.sigmoid(logits) for logits in task_logits]

            for i, task_name in enumerate(TASKS):
                all_labels[task_name].extend(labels[task_name].squeeze().tolist())
                all_preds[task_name].extend(task_preds[i].squeeze().cpu().tolist())

    auc_scores = {}
    ctr_task_name, cvr_task_name = TASKS[0], TASKS[1]
    
    auc_scores[f"{ctr_task_name}_auc"] = roc_auc_score(all_labels[ctr_task_name], all_preds[ctr_task_name])

    clicked_indices = np.where(np.array(all_labels[ctr_task_name]) == 1)[0]
    if len(clicked_indices) > 0:
        cvr_labels_clicked = np.array(all_labels[cvr_task_name])[clicked_indices]
        cvr_preds_clicked = np.array(all_preds[cvr_task_name])[clicked_indices]
        auc_scores[f"{cvr_task_name}_auc_post_click"] = roc_auc_score(cvr_labels_clicked, cvr_preds_clicked)
    else:
        auc_scores[f"{cvr_task_name}_auc_post_click"] = 0.0

    return auc_scores

def main():
    """
    主训练函数
    Main training function
    """
    # --- 1. 加载配置和数据 ---
    print("--- 🚀 开始M3oE模型训练流程 ---")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(FEATURE_INFO_PATH, 'r') as f:
        feature_max_idx = json.load(f)
    
    TAG_VOCAB_SIZE = COUNT_TAGS
    # 动态获取领域数量
    NUM_DOMAINS = COUNT_DOMIANS
    
    sparse_features = SPARSE_FEATURES
    # 使用安全、非泄露的特征列表
    dense_features = DENSE_FEATURES
    
    print(f"训练任务: {TASKS}, 领域数量: {NUM_DOMAINS}")

    # 创建数据加载器
    train_dataset = RecDataset(TRAIN_PKL_PATH, sparse_features, dense_features)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    test_dataset = RecDataset(TEST_PKL_PATH, sparse_features, dense_features)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 2. 初始化模型、优化器、损失函数 ---
    print("正在初始化M3oE模型...")
    EXPERT_HIDDEN_UNITS = HIDDEN_UNITS['expert']
    TOWER_HIDDEN_UNITS = HIDDEN_UNITS['tower']
    model = M3oE(
        sparse_feature_info=feature_max_idx,
        dense_feature_count=len(dense_features),
        emb_dim=EMBEDDING_DIM,
        rep_hidden_dim=REP_HIDDEN_DIM,
        expert_hidden_units=EXPERT_HIDDEN_UNITS,
        tower_hidden_units=TOWER_HIDDEN_UNITS,
        tag_vocab_size=TAG_VOCAB_SIZE,
        num_tasks=NUM_TASKS,
        num_domains=NUM_DOMAINS,
        num_shared_experts=NUM_SHARED_EXPERTS,
        domain_feature_name=DOMAIN_FEATURE_NAME
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss() 
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 3. 训练与评估循环 ---
    print("开始训练...")
    best_avg_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch in train_iterator:
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            tag_inputs = batch['tag_inputs'].to(device)
            labels = batch['labels']

            optimizer.zero_grad()
            task_logits = model(sparse_inputs, dense_inputs, tag_inputs)
            
            current_loss = 0
            for i, task_name in enumerate(TASKS):
                task_label = labels[task_name].to(device)
                loss = loss_fn(task_logits[i], task_label)
                current_loss += loss
            
            current_loss.backward()
            optimizer.step()
            total_loss += current_loss.item()
            
            train_iterator.set_postfix({
                "loss": f"{total_loss / (train_iterator.n + 1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # --- 在每个epoch结束后，进行评估 ---
        test_auc_scores = evaluate_m3oe_model(model, test_loader, device)
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
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_PATH, "m3oe_best_model.pth"))
            print(f"🎉 新的最优模型已保存! 平均AUC: {best_avg_auc:.4f}")

        scheduler.step()

    print(f"\n✅ 训练完成！最优模型已保存至 (最优平均AUC: {best_avg_auc:.4f})")

if __name__ == '__main__':
    main()
