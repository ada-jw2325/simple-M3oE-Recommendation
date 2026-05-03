# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import *

import os
from config import *
from datasets import RecDataset
from model import ESMM

import time
import datetime

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_ctr_labels, all_ctcvr_labels, all_cvr_labels = [], [], []
    all_ctr_preds, all_ctcvr_preds = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            
            tag_inputs = batch['tag_inputs'].to(device)
            p_ctr, p_ctcvr = model(sparse_inputs, dense_inputs, tag_inputs)

            all_ctr_labels.extend(batch['ctr_label'].squeeze().tolist())
            all_cvr_labels.extend(batch['cvr_label'].squeeze().tolist())
            all_ctcvr_labels.extend(batch['ctcvr_label'].squeeze().tolist())
            
            all_ctr_preds.extend(p_ctr.squeeze().cpu().tolist())
            all_ctcvr_preds.extend(p_ctcvr.squeeze().cpu().tolist())

    # 计算AUC
    ctr_auc = roc_auc_score(all_ctr_labels, all_ctr_preds)
    ctcvr_auc = roc_auc_score(all_ctcvr_labels, all_ctcvr_preds)
    
    # 计算pCVR的AUC (仅作为参考)
    # 筛选出点击为1的样本来评估CVR
    clicked_indices = [i for i, label in enumerate(all_ctr_labels) if label == 1]
    cvr_labels_clicked = np.array(all_cvr_labels)[clicked_indices]
    ctr_preds_clicked = np.array(all_ctr_preds)[clicked_indices]
    ctcvr_preds_clicked = np.array(all_ctcvr_preds)[clicked_indices]
    
    p_cvr = ctcvr_preds_clicked / (ctr_preds_clicked + 1e-8) # 加上一个很小的数防止除以0
    cvr_auc = roc_auc_score(cvr_labels_clicked, p_cvr)

    return ctr_auc, ctcvr_auc, cvr_auc


def trainAndEval(feature_max_idx, TAG_VOCAB_SIZE, dense_features, device, train_loader, test_loader):
    model = ESMM(
        sparse_feature_info = feature_max_idx,
        dense_feature_count = len(dense_features),
        emb_dim = EMBEDDING_DIM,
        hidden_units = HIDDEN_UNITS,
        tag_vocab_size=TAG_VOCAB_SIZE,
        max_tags=MAX_TAGS_LEN
    ).to(device)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.AdamW(model.parameters())
    loss = nn.BCELoss()
    # Learning rate update schedulers
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        train_iterator = tqdm(train_loader, desc=f"EPOCH {epoch+1}/{EPOCHS}")
        for batch in train_iterator:
            sparse_inputs = batch['sparse_inputs'].to(device)
            dense_inputs = batch['dense_inputs'].to(device)
            tag_inputs = batch['tag_inputs'].to(device)
            ctr_label = batch['ctr_label'].to(device)
            ctcvr_label = batch['ctcvr_label'].to(device)
            
            optimizer.zero_grad()
            p_ctr, p_ctcvr = model(sparse_inputs, dense_inputs, tag_inputs)
            loss_ctr = loss(p_ctr, ctr_label)
            loss_ctcvr = loss(p_ctcvr, ctcvr_label)
            loss_all = loss_ctcvr + loss_ctr

            train_loss.append(loss_all)

            loss_all.backward()
            optimizer.step()
        lr_scheduler.step()
        
        print(f"Epoch {epoch+1} finished. Evaluating on test set...")
        ctr_auc, ctcvr_auc, cvr_auc = evaluate_model(model, test_loader, device)
        log_message = f"Epoch {epoch+1}/{EPOCHS} - Test AUC: CTR={ctr_auc:.4f}, CVR={cvr_auc:.4f}, CTCVR={ctcvr_auc:.4f}"
        print(log_message)

        auc_avg = (ctr_auc + cvr_auc) / 2
        # 保存最优模型
        if auc_avg > best_auc:
            best_auc = auc_avg
            save_model_path = os.path.join(SAVED_MODEL_PATH , "esmm_best_model.pth")
            torch.save(model.state_dict(), save_model_path)
            print(f"Best model saved with AVG AUC: {auc_avg:.4f}")

def main():
    # --- 1. 加载数据和配置 ---
    print("Loading data and config...")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 打开稀疏特征表
    sparse_features_file_path = os.path.join(PROCESSED_DATA_PATH , 'feature_max_idx.json')
    with open(sparse_features_file_path, 'r') as f:
        feature_max_idx = json.load(f)

    TAG_VOCAB_SIZE = feature_max_idx['tagSize']
    feature_max_idx.pop('tagSize')


    sparse_features = SPARSE_FEATURES
    # 临时从测试集获取稠密特征数量（实际应从配置或预处理中获取）
    
    train_file_path = os.path.join(PROCESSED_DATA_PATH, "train_data.pkl")
    test_file_path = os.path.join(PROCESSED_DATA_PATH, "test_data.pkl")

    dense_features = DENSE_FEATURES
    
    train_dataset = RecDataset(train_file_path, sparse_features, dense_features)
    test_dataset = RecDataset(test_file_path, sparse_features, dense_features)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 2. 初始化模型、优化器、损失函数 ---
    print("Initializing model...")
    
    trainAndEval(feature_max_idx, TAG_VOCAB_SIZE, dense_features, device, train_loader, test_loader)

if __name__ == '__main__':
    main()