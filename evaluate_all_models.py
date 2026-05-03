import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ["ESMM", "MMOE", "MTL", "PLE", "M3OE"]


def run_eval(model_name: str) -> Path:
    model_dir = ROOT / model_name
    eval_script = model_dir / "eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"未找到评估脚本: {eval_script}")

    print(f"\n===== 运行 {model_name} 评估 =====")
    subprocess.run([sys.executable, str(eval_script)], cwd=str(model_dir), check=True)

    json_path = model_dir / f"{model_name.lower()}_evaluation_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"未找到评估结果 JSON: {json_path}")
    return json_path


def load_summary(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["model"], payload["results"]


def format_table(summaries):
    header = (
        "| Model | Split | CTR AUC | CVR AUC (Post Click) | Average AUC |\n"
        "|---|---:|---:|---:|---:|"
    )
    rows = [header]
    for model_name, results in summaries:
        for split in ["Online", "Random", "Combined"]:
            metrics = results[split]
            rows.append(
                f"| {model_name} | {split} | "
                f"{metrics['ctr_auc']:.4f} | {metrics['cvr_auc_post_click']:.4f} | {metrics['avg_auc']:.4f} |"
            )
    return "\n".join(rows)


def main() -> None:
    summaries = []
    for model_name in MODELS:
        json_path = run_eval(model_name)
        summaries.append(load_summary(json_path))

    table_md = format_table(summaries)

    out_md = ROOT / "evaluation_summary.md"
    out_json = ROOT / "evaluation_summary.json"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Multi-Model Evaluation Summary\n\n")
        f.write(table_md)
        f.write("\n")

    summary_payload = {
        model: results for model, results in summaries
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n===== 总评估完成 =====")
    print(f"Markdown 汇总: {out_md}")
    print(f"JSON 汇总: {out_json}")


if __name__ == "__main__":
    main()
