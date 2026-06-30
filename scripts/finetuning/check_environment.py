from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_ROWS = {
    "train": 13508,
    "valid": 673,
    "test": 776,
}
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


def check_jsonl(path: Path, expected_rows: int, errors: list[str]) -> None:
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
    if count != expected_rows:
        errors.append(f"{path.name} expected {expected_rows} rows, found {count}")


def check_project(project_root: Path, errors: list[str]) -> None:
    dataset_info = project_root / "data" / "dataset_info.json"
    if not dataset_info.exists():
        errors.append(f"missing dataset registry: {dataset_info}")
    else:
        print(f"[ok] dataset registry: {dataset_info}")

    data_dir = project_root / "data" / "synthetic_extractor"
    for split, expected_rows in EXPECTED_ROWS.items():
        check_jsonl(
            data_dir / f"MedDG_extractor_15k_{split}_alpaca.jsonl",
            expected_rows,
            errors,
        )

    for relative_path in (
        "configs/finetuning/extractor_qwen3_4b_lora.yaml",
        "configs/finetuning/extractor_qwen3_4b_predict.yaml",
        "configs/finetuning/extractor_qwen3_0_6b_lora.yaml",
        "configs/finetuning/extractor_qwen3_0_6b_predict.yaml",
    ):
        path = project_root / relative_path
        if not path.exists():
            errors.append(f"missing config: {path}")
        else:
            print(f"[ok] config: {relative_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the AskMed fine-tuning environment and datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only validate project files and datasets; skip CUDA and installed-package checks.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    errors: list[str] = []
    print(f"AskMed root: {project_root}")

    check_project(project_root, errors)
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
