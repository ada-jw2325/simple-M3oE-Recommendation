import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from datasets import RecDataset
from model import ESMM


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = {"ctr": [], "cvr": [], "ctcvr": []}
    all_preds = {"ctr": [], "ctcvr": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="正在评估", leave=False):
            sparse_inputs = batch["sparse_inputs"].to(device)
            dense_inputs = batch["dense_inputs"].to(device)
            tag_inputs = batch["tag_inputs"].to(device)

            p_ctr, p_ctcvr = model(sparse_inputs, dense_inputs, tag_inputs)

            all_labels["ctr"].extend(batch["ctr_label"].squeeze().tolist())
            all_labels["cvr"].extend(batch["cvr_label"].squeeze().tolist())
            all_labels["ctcvr"].extend(batch["ctcvr_label"].squeeze().tolist())
            all_preds["ctr"].extend(p_ctr.squeeze().cpu().tolist())
            all_preds["ctcvr"].extend(p_ctcvr.squeeze().cpu().tolist())

    ctr_auc = roc_auc_score(all_labels["ctr"], all_preds["ctr"])

    all_labels_np = {k: np.array(v) for k, v in all_labels.items()}
    all_preds_np = {k: np.array(v) for k, v in all_preds.items()}
    clicked_indices = np.where(all_labels_np["ctr"] == 1)[0]

    if len(clicked_indices) > 1:
        cvr_labels_clicked = all_labels_np["cvr"][clicked_indices]
        ctr_preds_clicked = all_preds_np["ctr"][clicked_indices]
        ctcvr_preds_clicked = all_preds_np["ctcvr"][clicked_indices]
        p_cvr = ctcvr_preds_clicked / (ctr_preds_clicked + 1e-8)
        if np.unique(cvr_labels_clicked).size > 1:
            cvr_auc_post_click = roc_auc_score(cvr_labels_clicked, p_cvr)
        else:
            cvr_auc_post_click = 0.5
    else:
        cvr_auc_post_click = 0.5

    avg_auc = float(np.mean([ctr_auc, cvr_auc_post_click]))
    return {
        "ctr_auc": float(ctr_auc),
        "cvr_auc_post_click": float(cvr_auc_post_click),
        "avg_auc": avg_auc,
    }


def main():
    print("--- 开始 ESMM 独立评估 ---")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    sparse_features_file_path = os.path.join(PROCESSED_DATA_PATH, "feature_max_idx.json")
    with open(sparse_features_file_path, "r") as f:
        feature_max_idx = json.load(f)

    tag_vocab_size = feature_max_idx["tagSize"]
    feature_max_idx.pop("tagSize")

    sparse_features = SPARSE_FEATURES
    dense_features = DENSE_FEATURES

    dataloaders = {
        "Online": DataLoader(
            RecDataset(os.path.join(PROCESSED_DATA_PATH, "online_test_data.pkl"), sparse_features, dense_features),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        "Random": DataLoader(
            RecDataset(os.path.join(PROCESSED_DATA_PATH, "random_test_data.pkl"), sparse_features, dense_features),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        "Combined": DataLoader(
            RecDataset(os.path.join(PROCESSED_DATA_PATH, "test_data.pkl"), sparse_features, dense_features),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
    }

    model_path = os.path.join(SAVED_MODEL_PATH, "esmm_best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    model = ESMM(
        sparse_feature_info=feature_max_idx,
        dense_feature_count=len(dense_features),
        emb_dim=EMBEDDING_DIM,
        hidden_units=HIDDEN_UNITS,
        tag_vocab_size=tag_vocab_size,
        max_tags=MAX_TAGS_LEN,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    results = {}
    for split_name, loader in dataloaders.items():
        print(f"评估数据集: {split_name}")
        results[split_name] = evaluate_model(model, loader, device)

    txt_path = os.path.join(ROOT_PATH, "esmm_evaluation_results.txt")
    json_path = os.path.join(ROOT_PATH, "esmm_evaluation_results.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("--- ESMM 模型评估结果 ---\n\n")
        for split_name in ["Online", "Random", "Combined"]:
            metrics = results[split_name]
            f.write(f"--- {split_name} ---\n")
            f.write(f"CTR_AUC: {metrics['ctr_auc']:.4f}\n")
            f.write(f"CVR_AUC_POST_CLICK: {metrics['cvr_auc_post_click']:.4f}\n")
            f.write(f"AVG_AUC: {metrics['avg_auc']:.4f}\n\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model": "ESMM", "results": results}, f, indent=2, ensure_ascii=False)

    print(f"评估完成，文本结果: {txt_path}")
    print(f"评估完成，JSON结果: {json_path}")


if __name__ == "__main__":
    main()
