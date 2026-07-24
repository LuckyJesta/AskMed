from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"system", "instruction", "input", "output"}
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "modelscope",
    "yaml",
    "tensorboard",
)

def check_cli(errors: list[str]) -> None:
    executable = shutil.which("llamafactory-cli")
    if executable is None:
        errors.append("llamafactory-cli was not found in PATH")
        return

    completed = subprocess.run(
        [executable, "version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        errors.append(f"llamafactory-cli version failed: {output}")
    else:
        print(f"[ok] LLaMA-Factory: {output}")


def check_packages(errors: list[str]) -> Any:
    imported: dict[str, Any] = {}
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            errors.append(f"cannot import {package}: {exc}")
            continue
        imported[package] = module
        version = getattr(module, "__version__", "unknown")
        print(f"[ok] {package}: {version}")
    return imported.get("torch")


def check_gpu(torch_module: Any, errors: list[str]) -> None:
    if torch_module is None:
        return
    try:
        if not torch_module.cuda.is_available():
            errors.append("CUDA is not available")
            return
        device = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device)
    except RuntimeError as exc:
        errors.append(f"CUDA initialization failed: {exc}")
        return

    memory_gb = properties.total_memory / 1024**3
    print(f"[ok] GPU: {properties.name}")
    print(f"[ok] GPU memory: {memory_gb:.1f} GB")
    if memory_gb < 22:
        print("[warn] Less than 22 GB VRAM: cutoff_len=8192 may require a shorter fallback.")

    bf16_supported = bool(torch_module.cuda.is_bf16_supported())
    print(f"[ok] BF16 supported: {bf16_supported}")
    if not bf16_supported:
        errors.append("GPU does not report BF16 support")


def check_jsonl(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing dataset: {path}")
        return

    count = 0
    for line_no, line in enumerate(path.open("r", encoding="utf-8"), 1):
        if not line.strip():
            continue
        count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no} invalid JSON: {exc}")
            continue
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{path}:{line_no} missing fields: {sorted(missing)}")
        if "meddg_weak_labels" in str(row.get("input", "")):
            errors.append(f"{path}:{line_no} contains meddg_weak_labels")

    print(f"[ok] {path.name}: {count} rows")


def read_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing config: {path}")
        return {}
    try:
        import yaml
    except Exception as exc:
        errors.append(f"cannot import yaml while reading {path}: {exc}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        errors.append(f"cannot read config {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"config must be a YAML object: {path}")
        return {}
    print(f"[ok] config: {path.relative_to(path.parents[2]) if len(path.parents) > 2 else path}")
    return data


def split_dataset_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def check_dataset_key(project_root: Path, registry: dict[str, Any], dataset_key: str, errors: list[str]) -> None:
    dataset_row = registry.get(dataset_key)
    if not dataset_row:
        errors.append(f"missing dataset key in registry: {dataset_key}")
        return
    file_name = dataset_row.get("file_name")
    if not file_name:
        errors.append(f"dataset key {dataset_key} has no file_name")
        return
    check_jsonl(project_root / "data" / str(file_name), errors)


def check_project(
    project_root: Path,
    train_config: Path | None,
    predict_config: Path | None,
    gold_dataset: Path | None,
    errors: list[str],
) -> None:
    dataset_info = project_root / "data" / "dataset_info.json"
    if not dataset_info.exists():
        errors.append(f"missing dataset registry: {dataset_info}")
        return
    print(f"[ok] dataset registry: {dataset_info}")
    registry = json.loads(dataset_info.read_text(encoding="utf-8"))
    checked: set[str] = set()

    if train_config is not None:
        config = read_yaml(train_config, errors)
        for key in split_dataset_names(config.get("dataset")) + split_dataset_names(config.get("eval_dataset")):
            if key not in checked:
                check_dataset_key(project_root, registry, key, errors)
                checked.add(key)

    if predict_config is not None:
        config = read_yaml(predict_config, errors)
        for key in split_dataset_names(config.get("dataset")) + split_dataset_names(config.get("eval_dataset")):
            if key not in checked:
                check_dataset_key(project_root, registry, key, errors)
                checked.add(key)

    if gold_dataset is not None:
        check_jsonl(gold_dataset, errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the AskMed fine-tuning environment and datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only validate project files and datasets; skip CUDA and installed-package checks.",
    )
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--predict-config", type=Path)
    parser.add_argument("--gold", type=Path, help="Optional gold Alpaca JSONL used for structured evaluation.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    errors: list[str] = []
    print(f"AskMed root: {project_root}")

    train_config = (project_root / args.train_config).resolve() if args.train_config else None
    predict_config = (project_root / args.predict_config).resolve() if args.predict_config else None
    gold_dataset = (project_root / args.gold).resolve() if args.gold else None
    check_project(project_root, train_config, predict_config, gold_dataset, errors)
    if not args.data_only:
        check_cli(errors)
        torch_module = check_packages(errors)
        check_gpu(torch_module, errors)

    if errors:
        print("\nEnvironment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    print("\nAll requested checks passed.")


if __name__ == "__main__":
    main()
