import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from datasets import RecDataset
from model import PLE


def evaluate_model(model, dataloader, device):
	model.eval()
	all_labels = {task: [] for task in TASKS}
	all_preds = {task: [] for task in TASKS}

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="正在评估", leave=False):
			sparse_inputs = batch["sparse_inputs"].to(device)
			dense_inputs = batch["dense_inputs"].to(device)
			tag_inputs = batch["tag_inputs"].to(device)
			labels = batch["labels"]

			task_probs = model(sparse_inputs, dense_inputs, tag_inputs)

			for i, task_name in enumerate(TASKS):
				all_labels[task_name].extend(labels[task_name].squeeze().tolist())
				all_preds[task_name].extend(task_probs[i].squeeze().cpu().tolist())

	ctr_task_name, cvr_task_name = TASKS[0], TASKS[1]
	ctr_auc = roc_auc_score(all_labels[ctr_task_name], all_preds[ctr_task_name])

	clicked_indices = np.where(np.array(all_labels[ctr_task_name]) == 1)[0]
	if len(clicked_indices) > 1:
		cvr_labels_clicked = np.array(all_labels[cvr_task_name])[clicked_indices]
		cvr_preds_clicked = np.array(all_preds[cvr_task_name])[clicked_indices]
		if np.unique(cvr_labels_clicked).size > 1:
			cvr_auc_post_click = roc_auc_score(cvr_labels_clicked, cvr_preds_clicked)
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
	print("--- 开始 PLE 独立评估 ---")
	device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
	print(f"使用设备: {device}")

	with open(FEATURE_INFO_PATH, "r") as f:
		feature_max_idx = json.load(f)

	sparse_features = SPARSE_FEATURES
	dense_features = DENSE_FEATURES

	dataloaders = {
		"Online": DataLoader(
			RecDataset(ONLINE_TEST_PKL_PATH, sparse_features, dense_features),
			batch_size=BATCH_SIZE,
			shuffle=False,
			num_workers=0,
		),
		"Random": DataLoader(
			RecDataset(RANDOM_TEST_PKL_PATH, sparse_features, dense_features),
			batch_size=BATCH_SIZE,
			shuffle=False,
			num_workers=0,
		),
		"Combined": DataLoader(
			RecDataset(TEST_PKL_PATH, sparse_features, dense_features),
			batch_size=BATCH_SIZE,
			shuffle=False,
			num_workers=0,
		),
	}

	model_path = os.path.join(SAVED_MODEL_PATH, "ple_best_model.pth")
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"未找到模型文件: {model_path}")

	model = PLE(
		sparse_feature_info=feature_max_idx,
		dense_feature_count=len(dense_features),
		emb_dim=EMBEDDING_DIM,
		cgc_hidden_units=HIDDEN_UNITS,
		tag_vocab_size=COUNT_TAGS,
		num_tasks=len(TASKS),
		num_shared_experts=NUM_SHARED_EXPERTS,
		num_task_experts=NUM_TASK_EXPERTS,
	).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))

	results = {}
	for split_name, loader in dataloaders.items():
		print(f"评估数据集: {split_name}")
		results[split_name] = evaluate_model(model, loader, device)

	txt_path = os.path.join(ROOT_PATH, "ple_evaluation_results.txt")
	json_path = os.path.join(ROOT_PATH, "ple_evaluation_results.json")

	with open(txt_path, "w", encoding="utf-8") as f:
		f.write("--- PLE 模型评估结果 ---\n\n")
		for split_name in ["Online", "Random", "Combined"]:
			metrics = results[split_name]
			f.write(f"--- {split_name} ---\n")
			f.write(f"CTR_AUC: {metrics['ctr_auc']:.4f}\n")
			f.write(f"CVR_AUC_POST_CLICK: {metrics['cvr_auc_post_click']:.4f}\n")
			f.write(f"AVG_AUC: {metrics['avg_auc']:.4f}\n\n")

	with open(json_path, "w", encoding="utf-8") as f:
		json.dump({"model": "PLE", "results": results}, f, indent=2, ensure_ascii=False)

	print(f"评估完成，文本结果: {txt_path}")
	print(f"评估完成，JSON结果: {json_path}")


if __name__ == "__main__":
	main()
