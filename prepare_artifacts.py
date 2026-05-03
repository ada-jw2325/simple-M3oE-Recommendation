import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def run_script(script_path: Path, cwd: Path) -> None:
    print(f"运行脚本: {script_path} (cwd={cwd})")
    subprocess.run([sys.executable, str(script_path)], cwd=str(cwd), check=True)


def ensure_processed_feature_dict_alignment() -> None:
    processed_dir = DATA_DIR / "processed"
    full_dict_path = processed_dir / "feature_max_idx.json"
    without_tag_path = processed_dir / "feature_max_idx_withoutTag.json"

    if full_dict_path.exists() and not without_tag_path.exists():
        with open(full_dict_path, "r", encoding="utf-8") as f:
            full_dict = json.load(f)
        without_tag = {k: v for k, v in full_dict.items() if k != "tagSize"}
        with open(without_tag_path, "w", encoding="utf-8") as f:
            json.dump(without_tag, f)
        print(f"已生成: {without_tag_path}")


def verify_artifacts() -> None:
    required = [
        DATA_DIR / "processed" / "train_data.pkl",
        DATA_DIR / "processed" / "test_data.pkl",
        DATA_DIR / "processed" / "online_test_data.pkl",
        DATA_DIR / "processed" / "random_test_data.pkl",
        DATA_DIR / "processed" / "feature_max_idx.json",
        DATA_DIR / "processed" / "feature_max_idx_withoutTag.json",
        DATA_DIR / "processedM3OE" / "train_data.pkl",
        DATA_DIR / "processedM3OE" / "test_data.pkl",
        DATA_DIR / "processedM3OE" / "online_test_data.pkl",
        DATA_DIR / "processedM3OE" / "random_test_data.pkl",
        DATA_DIR / "processedM3OE" / "feature_max_idx_withoutTag.json",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("以下产物缺失:\n" + "\n".join(missing))

    print("数据与特征字典产物检查通过。")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一生成并校验多模型训练/评估产物")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过预处理，仅做字典对齐与产物校验",
    )
    args = parser.parse_args()

    if not args.skip_preprocess:
        run_script(ROOT / "ESMM" / "preprocess.py", ROOT / "ESMM")
        run_script(ROOT / "M3OE" / "preprocess.py", ROOT / "M3OE")

    ensure_processed_feature_dict_alignment()
    verify_artifacts()
    print("全部完成。")


if __name__ == "__main__":
    main()
