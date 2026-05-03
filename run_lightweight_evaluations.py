import importlib
import json
import os
import pickle
import runpy
import sys
import types
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SMALL_N = 5000


def sample_pkl(src_path, dst_path, n=SMALL_N):
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "sample") and hasattr(data, "__len__") and len(data) > n:
        sampled = data.sample(n=n, random_state=42)
    elif hasattr(data, "__len__") and len(data) > n:
        sampled = data[:n]
    else:
        sampled = data

    with open(dst_path, "wb") as f:
        pickle.dump(sampled, f)


def ensure_small_datasets():
    processed = os.path.join(PROJECT_ROOT, "data", "processed")
    processed_small = os.path.join(PROJECT_ROOT, "data", "processed_small")
    os.makedirs(processed_small, exist_ok=True)

    # Base train/test for lightweight training/eval
    sample_pkl(os.path.join(processed, "train_data.pkl"), os.path.join(processed_small, "train_data.pkl"))
    sample_pkl(os.path.join(processed, "test_data.pkl"), os.path.join(processed_small, "test_data.pkl"))
    sample_pkl(os.path.join(processed, "online_test_data.pkl"), os.path.join(processed_small, "online_test_data.pkl"))
    sample_pkl(os.path.join(processed, "random_test_data.pkl"), os.path.join(processed_small, "random_test_data.pkl"))

    for fn in ["feature_max_idx.json", "feature_max_idx_withoutTag.json"]:
        src = os.path.join(processed, fn)
        dst = os.path.join(processed_small, fn)
        if os.path.exists(src):
            with open(src, "rb") as fsrc:
                raw = fsrc.read()
            with open(dst, "wb") as fdst:
                fdst.write(raw)

    processed_m3oe = os.path.join(PROJECT_ROOT, "data", "processedM3OE")
    processed_m3oe_small = os.path.join(PROJECT_ROOT, "data", "processedM3OE_small")
    os.makedirs(processed_m3oe_small, exist_ok=True)

    sample_pkl(os.path.join(processed_m3oe, "train_data.pkl"), os.path.join(processed_m3oe_small, "train_data.pkl"))
    sample_pkl(os.path.join(processed_m3oe, "test_data.pkl"), os.path.join(processed_m3oe_small, "test_data.pkl"))
    sample_pkl(os.path.join(processed_m3oe, "online_test_data.pkl"), os.path.join(processed_m3oe_small, "online_test_data.pkl"))
    sample_pkl(os.path.join(processed_m3oe, "random_test_data.pkl"), os.path.join(processed_m3oe_small, "random_test_data.pkl"))

    src = os.path.join(processed_m3oe, "feature_max_idx_withoutTag.json")
    dst = os.path.join(processed_m3oe_small, "feature_max_idx_withoutTag.json")
    if os.path.exists(src):
        with open(src, "rb") as fsrc:
            raw = fsrc.read()
        with open(dst, "wb") as fdst:
            fdst.write(raw)


def run_eval(model_dir, package_name, overrides):
    cfg_path = os.path.join(PROJECT_ROOT, model_dir, "config.py")
    cfg = runpy.run_path(cfg_path)

    cfg_mod = types.ModuleType("config")
    cfg_mod.__dict__.update(cfg)
    cfg_mod.__dict__.update(overrides)

    sys.modules["config"] = cfg_mod
    for stale in ["datasets", "model", "eval"]:
        if stale in sys.modules:
            del sys.modules[stale]

    model_path = os.path.join(PROJECT_ROOT, model_dir)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    eval_module = importlib.import_module(f"{package_name}.eval")
    importlib.reload(eval_module)
    eval_module.main()


def load_eval_json(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(results):
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "lightweight_retrain_cpu",
        "dataset": {
            "train_eval": "processed_small/processedM3OE_small sampled 5000 rows",
            "splits": ["Online", "Random", "Combined"],
            "cvr_metric": "CVR_AUC_POST_CLICK",
        },
        "models": results,
    }

    combined_rank = []
    for model_name, payload in results.items():
        combined = payload.get("results", {}).get("Combined", {})
        combined_rank.append(
            {
                "model": model_name,
                "avg_auc": combined.get("avg_auc", 0.0),
                "ctr_auc": combined.get("ctr_auc", 0.0),
                "cvr_auc_post_click": combined.get("cvr_auc_post_click", 0.0),
            }
        )

    combined_rank.sort(key=lambda x: x["avg_auc"], reverse=True)
    summary["combined_ranking"] = combined_rank
    return summary


def write_summary_files(summary):
    json_path = os.path.join(PROJECT_ROOT, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(PROJECT_ROOT, "evaluation_summary.md")
    lines = []
    lines.append("# 轻量化重训练评估汇总")
    lines.append("")
    lines.append(f"- 生成时间: {summary['generated_at']}")
    lines.append(f"- 运行模式: {summary['mode']}")
    lines.append("- 说明: 本次结果来自 CPU + 小样本快速重训练，用于替换来源不明的 checkpoint 并验证流程可复现。")
    lines.append("")
    lines.append("## Combined 集排名（按 AVG_AUC）")
    lines.append("")
    lines.append("| Rank | Model | AVG_AUC | CTR_AUC | CVR_AUC_POST_CLICK |")
    lines.append("|---|---|---:|---:|---:|")
    for i, item in enumerate(summary["combined_ranking"], start=1):
        lines.append(
            f"| {i} | {item['model']} | {item['avg_auc']:.4f} | {item['ctr_auc']:.4f} | {item['cvr_auc_post_click']:.4f} |"
        )

    lines.append("")
    lines.append("## 分模型分数据集指标")
    lines.append("")
    for model_name, payload in summary["models"].items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Split | CTR_AUC | CVR_AUC_POST_CLICK | AVG_AUC |")
        lines.append("|---|---:|---:|---:|")
        for split in ["Online", "Random", "Combined"]:
            m = payload["results"][split]
            lines.append(
                f"| {split} | {m['ctr_auc']:.4f} | {m['cvr_auc_post_click']:.4f} | {m['avg_auc']:.4f} |"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ensure_small_datasets()

    checkpoints = os.path.join(PROJECT_ROOT, "checkpoints")
    processed_small = os.path.join(PROJECT_ROOT, "data", "processed_small")
    processed_m3oe_small = os.path.join(PROJECT_ROOT, "data", "processedM3OE_small")

    # ESMM
    run_eval(
        "ESMM",
        "ESMM",
        {
            "PROCESSED_DATA_PATH": processed_small,
            "SAVED_MODEL_PATH": checkpoints,
            "ROOT_PATH": PROJECT_ROOT,
            "BATCH_SIZE": 64,
            "DEVICE": "cpu",
            "EMBEDDING_DIM": 8,
            "HIDDEN_UNITS": [128, 32, 8],
        },
    )

    # MTL
    run_eval(
        "MTL",
        "MTL",
        {
            "SAVED_MODEL_PATH": checkpoints,
            "ROOT_PATH": PROJECT_ROOT,
            "FEATURE_INFO_PATH": os.path.join(processed_small, "feature_max_idx_withoutTag.json"),
            "ONLINE_TEST_PKL_PATH": os.path.join(processed_small, "online_test_data.pkl"),
            "RANDOM_TEST_PKL_PATH": os.path.join(processed_small, "random_test_data.pkl"),
            "TEST_PKL_PATH": os.path.join(processed_small, "test_data.pkl"),
            "BATCH_SIZE": 64,
            "DEVICE": "cpu",
            "EMBEDDING_DIM": 8,
        },
    )

    # MMOE
    run_eval(
        "MMOE",
        "MMOE",
        {
            "SAVED_MODEL_PATH": checkpoints,
            "ROOT_PATH": PROJECT_ROOT,
            "FEATURE_INFO_PATH": os.path.join(processed_small, "feature_max_idx_withoutTag.json"),
            "ONLINE_TEST_PKL_PATH": os.path.join(processed_small, "online_test_data.pkl"),
            "RANDOM_TEST_PKL_PATH": os.path.join(processed_small, "random_test_data.pkl"),
            "TEST_PKL_PATH": os.path.join(processed_small, "test_data.pkl"),
            "BATCH_SIZE": 64,
            "DEVICE": "cpu",
            "EMBEDDING_DIM": 8,
        },
    )

    # PLE
    run_eval(
        "PLE",
        "PLE",
        {
            "SAVED_MODEL_PATH": checkpoints,
            "ROOT_PATH": PROJECT_ROOT,
            "FEATURE_INFO_PATH": os.path.join(processed_small, "feature_max_idx_withoutTag.json"),
            "ONLINE_TEST_PKL_PATH": os.path.join(processed_small, "online_test_data.pkl"),
            "RANDOM_TEST_PKL_PATH": os.path.join(processed_small, "random_test_data.pkl"),
            "TEST_PKL_PATH": os.path.join(processed_small, "test_data.pkl"),
            "BATCH_SIZE": 64,
            "DEVICE": "cpu",
            "EMBEDDING_DIM": 8,
        },
    )

    # M3OE
    run_eval(
        "M3OE",
        "M3OE",
        {
            "SAVED_MODEL_PATH": checkpoints,
            "ROOT_PATH": PROJECT_ROOT,
            "FEATURE_INFO_PATH": os.path.join(processed_m3oe_small, "feature_max_idx_withoutTag.json"),
            "ONLINE_TEST_PKL_PATH": os.path.join(processed_m3oe_small, "online_test_data.pkl"),
            "RANDOM_TEST_PKL_PATH": os.path.join(processed_m3oe_small, "random_test_data.pkl"),
            "TEST_PKL_PATH": os.path.join(processed_m3oe_small, "test_data.pkl"),
            "BATCH_SIZE": 64,
            "DEVICE": "cpu",
            "EMBEDDING_DIM": 8,
        },
    )

    results = {
        "ESMM": load_eval_json("esmm_evaluation_results.json"),
        "MTL": load_eval_json("mtl_evaluation_results.json"),
        "MMOE": load_eval_json("mmoe_evaluation_results.json"),
        "PLE": load_eval_json("ple_evaluation_results.json"),
        "M3OE": load_eval_json("m3oe_evaluation_results.json"),
    }

    summary = build_summary(results)
    write_summary_files(summary)

    print("All lightweight evaluations finished.")
    print("Wrote: evaluation_summary.json")
    print("Wrote: evaluation_summary.md")


if __name__ == "__main__":
    main()
