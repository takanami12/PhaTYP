#!/usr/bin/env python3
"""
K-Fold Fine-tuning for PhaTYP model.

PIPELINE:
  1. Collect all unique genomes across all folds/groups
  2. Run preprocessing ONCE (Prodigal + DIAMOND) -> PC token features
  3. For each fold × group: fine-tune a separate model, evaluate

EXPECTED DATA STRUCTURE:
  data_dir/
  ├── fold_0/
  │   ├── groupA_train.fasta
  │   ├── groupA_val.fasta
  │   ├── groupB_train.fasta
  │   ├── groupB_val.fasta
  │   └── ...
  ├── fold_1/
  │   └── ...
  └── fold_K/
      └── ...

LABEL CSV (genome_id,label,source):
  genome_id   : sequence ID matching FASTA header
  label       : 0=temperate, 1=virulent
  source      : FASTA filename containing this genome (for reference only)

USAGE:
  # Run from PhaTYP root directory:
  python train/run_kfold_finetune.py \
      --data_dir  /path/to/data \
      --label_csv /path/to/labels.csv \
      --pretrained_model train/log_dapt \
      --output_dir kfold_results

  # Multi-GPU (e.g., 4 GPUs):
  torchrun --nproc_per_node=4 train/run_kfold_finetune.py \
      --data_dir  /path/to/data \
      --label_csv /path/to/labels.csv \
      --pretrained_model train/log_dapt \
      --output_dir kfold_results
"""

import os
import re
import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import datasets
from datasets import Dataset, DatasetDict

import transformers
from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from Bio import SeqIO

# Suppress verbose progress bars from datasets and transformers
datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="K-Fold Fine-tuning for PhaTYP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory containing fold_X subdirectories with FASTA files",
    )
    parser.add_argument(
        "--label_csv", required=True,
        help="CSV file with columns: genome_id, label, source "
             "(label: 1=virulent, 0=temperate)",
    )
    parser.add_argument(
        "--pretrained_model", default="train/log_dapt",
        help="Pretrained PhaTYP model directory (output of continued_pretrain.py). "
             "Must be run from PhaTYP root directory.",
    )
    # Default: parent of this script (train/../ == PhaTYP root)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_phatyp_dir = os.path.dirname(_script_dir)
    parser.add_argument(
        "--config_dir", default=os.path.join(_default_phatyp_dir, "config"),
        help="BERT config/tokenizer directory (PhaTYP root/config)",
    )
    parser.add_argument(
        "--output_dir", default="kfold_results",
        help="Output directory for models, logs, and results",
    )
    parser.add_argument(
        "--phatyp_dir", default=_default_phatyp_dir,
        help="PhaTYP root directory (contains preprocessing.py and database/). "
             "Default: parent directory of this script.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of fine-tuning epochs per fold",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Per-device batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--min_len", type=int, default=0,
        help="Minimum contig length (bp) to keep during preprocessing "
             "(default 0 = keep all). PhaTYP's original default is 3000, "
             "but complete phage genomes are often shorter.",
    )
    parser.add_argument(
        "--threads", type=str, default="8",
        help="Number of threads for DIAMOND and Prodigal",
    )
    parser.add_argument(
        "--skip_preprocessing", action="store_true",
        help="Skip preprocessing step and load cached features from output_dir/preprocessing/",
    )
    args = parser.parse_args()
    args.pretrained_model = str(Path(args.pretrained_model).resolve())
    args.config_dir = str(Path(args.config_dir).resolve())
    return args


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def collect_fasta_files(data_dir: Path):
    """Return sorted list of all FASTA files under data_dir/fold_*/ directories."""
    fasta_files = []
    for fold_dir in sorted(data_dir.iterdir()):
        if fold_dir.is_dir() and "fold" in fold_dir.name.lower():
            for ext in ("*.fasta", "*.fa", "*.fna"):
                fasta_files.extend(sorted(fold_dir.glob(ext)))
    return fasta_files


def get_genome_ids(fasta_path: Path):
    """Return list of genome IDs from a FASTA file (first part of header split by '__')."""
    return [r.id.split("__")[0] for r in SeqIO.parse(str(fasta_path), "fasta")]


# ---------------------------------------------------------------------------
# Step 1 – Preprocessing (run once for all unique genomes)
# ---------------------------------------------------------------------------

def run_preprocessing(fasta_files, phatyp_dir, midfolder, min_len, threads):
    """
    Collect all unique genomes from fasta_files, run preprocessing.py once,
    and return a dict mapping genome_id -> PC token text string.
    """
    os.makedirs(midfolder, exist_ok=True)

    # Collect unique sequences that pass the length filter
    seen_ids = set()
    all_records = []
    short_skipped = 0

    for fasta_path in fasta_files:
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            if record.id in seen_ids:
                continue
            if len(record.seq) < min_len:
                short_skipped += 1
                seen_ids.add(record.id)   # mark as seen so we don't warn twice
                continue
            seen_ids.add(record.id)
            all_records.append(record)

    print(f"  Unique genomes passing length filter (>={min_len} bp): {len(all_records)}")
    if short_skipped:
        print(f"  Skipped {short_skipped} sequences shorter than {min_len} bp")

    if len(all_records) == 0:
        raise RuntimeError(
            f"No sequences passed the length filter (min_len={min_len} bp). "
            f"All {short_skipped} sequences were shorter than {min_len} bp.\n"
            f"  -> Re-run with --min_len 0 to keep all sequences, or set a smaller value."
        )

    # Write combined FASTA
    combined_fasta = os.path.join(midfolder, "combined_all.fasta")
    SeqIO.write(all_records, combined_fasta, "fasta")

    # Run preprocessing.py (length filter already applied, set --len 0)
    preprocess_script = os.path.join(phatyp_dir, "preprocessing.py")
    cmd = [
        sys.executable, preprocess_script,
        "--contigs", combined_fasta,
        "--midfolder", midfolder,
        "--len", "0",          # skip built-in filter (we already filtered above)
        "--threads", str(threads),
    ]
    print(f"  Running: {' '.join(cmd)}")
    log_path = os.path.join(midfolder, "preprocessing.log")
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(cmd, check=True, cwd=phatyp_dir, stdout=log_f, stderr=log_f)
    except subprocess.CalledProcessError as e:
        # Print the last 50 lines of the log so the user can diagnose the error
        print(f"\n[ERROR] preprocessing.py failed with exit code {e.returncode}")
        print(f"  Log file: {log_path}")
        print("  --- Last 50 lines of preprocessing log ---")
        try:
            with open(log_path) as _lf:
                tail = _lf.readlines()
            for line in tail[-50:]:
                print("  " + line, end="")
        except Exception:
            print("  (could not read log file)")
        print("  --- End of log ---\n")
        raise
    print(f"  Preprocessing log saved to: {log_path}")

    return _load_features(midfolder)


def _load_features(midfolder):
    """Load bert_feat.csv + id2contig dict and return genome_id -> text dict."""
    bert_feat_path = os.path.join(midfolder, "bert_feat.csv")
    id2contig_path = os.path.join(midfolder, "sentence_id2contig.dict")

    if not os.path.exists(bert_feat_path):
        raise FileNotFoundError(f"bert_feat.csv not found in {midfolder}")
    if not os.path.exists(id2contig_path):
        raise FileNotFoundError(f"sentence_id2contig.dict not found in {midfolder}")

    feat_df = pd.read_csv(bert_feat_path)
    with open(id2contig_path, "rb") as f:
        id2contig = pickle.load(f)

    genome2text = {}
    for idx, text in enumerate(feat_df["text"].values):
        genome_id = id2contig[idx].split("__")[0]
        genome2text[genome_id] = text

    print(f"  Loaded features for {len(genome2text)} genomes")
    return genome2text


# ---------------------------------------------------------------------------
# Step 2 – Build per-fold train/val DataFrames
# ---------------------------------------------------------------------------

def build_fold_dataframes(fold_dir: Path, label_map: dict, genome2text: dict):
    """
    Build train and val DataFrames for each group in one fold separately.

    Group name is extracted from the FASTA filename by removing the
    trailing '_train' or '_val' suffix (e.g. 'groupA_train.fasta' -> 'groupA').

    Returns:
        dict mapping group_name -> (train_df, val_df)  (columns: label, text)
    """
    groups = {}   # group_name -> {"train": [...], "val": [...]}
    missing_feat, missing_label = [], []

    fasta_files = []
    for ext in ("*.fasta", "*.fa", "*.fna"):
        fasta_files.extend(sorted(fold_dir.glob(ext)))

    for fasta_path in fasta_files:
        name_lower = fasta_path.name.lower()
        if "train" in name_lower:
            split = "train"
        elif "val" in name_lower:
            split = "val"
        else:
            print(f"  [WARN] Cannot determine split for '{fasta_path.name}' — skipping")
            continue

        group = re.sub(r'_(train|val)$', '', fasta_path.stem, flags=re.IGNORECASE)
        if group not in groups:
            groups[group] = {"train": [], "val": []}

        for genome_id in get_genome_ids(fasta_path):
            if genome_id not in genome2text:
                missing_feat.append(genome_id)
                continue
            if genome_id not in label_map:
                missing_label.append(genome_id)
                continue
            record = {"label": label_map[genome_id], "text": genome2text[genome_id]}
            groups[group][split].append(record)

    if missing_feat:
        print(f"  [WARN] {len(missing_feat)} genomes skipped (no features — likely too short)")
    if missing_label:
        print(f"  [WARN] {len(missing_label)} genomes skipped (no label in label CSV)")

    return {
        group: (pd.DataFrame(splits["train"]), pd.DataFrame(splits["val"]))
        for group, splits in groups.items()
    }


# ---------------------------------------------------------------------------
# Step 3 – Fine-tune + evaluate one fold
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute per-class accuracy and overall accuracy.
    Label convention: 0=temperate, 1=virulent
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    viru_acc    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    temp_acc    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    overall_acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "virulent_acc":  round(viru_acc, 4),
        "temperate_acc": round(temp_acc, 4),
        "overall_acc":   round(overall_acc, 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def finetune_fold(fold_name, fold_idx, group_name, train_df, val_df, args, tokenizer):
    """Fine-tune model for one fold/group and return evaluation metrics."""
    fold_out = os.path.join(args.output_dir, fold_name, group_name)
    os.makedirs(fold_out, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx} ({fold_name}) — GROUP {group_name}")
    print(f"  Train: {len(train_df)} samples  "
          f"| virulent={int((train_df['label']==1).sum())} "
          f"| temperate={int((train_df['label']==0).sum())}")
    print(f"  Val:   {len(val_df)} samples  "
          f"| virulent={int((val_df['label']==1).sum())} "
          f"| temperate={int((val_df['label']==0).sum())}")
    print(f"{'=' * 60}")

    # Save split CSVs for reproducibility
    train_df.to_csv(os.path.join(fold_out, "train.csv"), index=False)
    val_df.to_csv(os.path.join(fold_out, "val.csv"), index=False)

    # Build HuggingFace DatasetDict
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True)

    ds_train = Dataset(pa.Table.from_pandas(train_df.reset_index(drop=True)))
    ds_val   = Dataset(pa.Table.from_pandas(val_df.reset_index(drop=True)))
    data     = DatasetDict({"train": ds_train, "test": ds_val})
    tok_data = data.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Fresh model for each fold (loaded from pretrained checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(fold_out, "checkpoints"),
        do_train=True,
        do_eval=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_acc",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",          # set to "tensorboard" if you want TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_data["train"],
        eval_dataset=tok_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final prediction on val set
    pred_out = trainer.predict(tok_data["test"])
    metrics  = compute_metrics((pred_out.predictions, pred_out.label_ids))

    print(f"\n  [Fold {fold_idx}] virulent_acc={metrics['virulent_acc']:.4f} "
          f"| temperate_acc={metrics['temperate_acc']:.4f} "
          f"| overall_acc={metrics['overall_acc']:.4f}")
    print(f"             TP={metrics['TP']} TN={metrics['TN']} "
          f"FP={metrics['FP']} FN={metrics['FN']}")

    # Save final model and metrics
    trainer.save_model(os.path.join(fold_out, "model"))
    with open(os.path.join(fold_out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load label CSV ----
    print("\n[1] Loading label CSV...")
    label_df = pd.read_csv(args.label_csv)
    missing = {"genome_id", "label", "source"} - set(label_df.columns)
    if missing:
        raise ValueError(f"label_csv is missing columns: {missing}")
    label_map = dict(zip(label_df["genome_id"], label_df["label"]))
    print(f"    Loaded labels for {len(label_map)} genomes")
    print(f"    Label distribution: {label_df['label'].value_counts().to_dict()}")

    # ---- Collect FASTA files ----
    data_dir = Path(args.data_dir)
    print("\n[2] Collecting FASTA files...")
    fasta_files = collect_fasta_files(data_dir)
    fold_dirs   = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and "fold" in d.name.lower()]
    )
    print(f"    Found {len(fasta_files)} FASTA files across {len(fold_dirs)} folds")
    for fd in fold_dirs:
        print(f"      {fd.name}/")

    if not fasta_files:
        raise RuntimeError(f"No FASTA files found under {data_dir}. "
                           "Check --data_dir and that fold directories exist.")

    # ---- Preprocessing ----
    preprocess_dir = os.path.join(args.output_dir, "preprocessing")
    if args.skip_preprocessing:
        print(f"\n[3] Skipping preprocessing — loading cached features from {preprocess_dir}")
        genome2text = _load_features(preprocess_dir)
    else:
        print("\n[3] Running preprocessing (Prodigal + DIAMOND) on all unique genomes...")
        print("    This step can take several hours for large datasets.")
        genome2text = run_preprocessing(
            fasta_files, args.phatyp_dir, preprocess_dir, args.min_len, args.threads
        )

    # ---- Load tokenizer ----
    tokenizer = BertTokenizer.from_pretrained(args.config_dir, do_basic_tokenize=False)

    # ---- K-Fold fine-tuning ----
    print(f"\n[4] Starting K-Fold fine-tuning ({len(fold_dirs)} folds, per-group)...")
    all_metrics = []

    for fold_idx, fold_dir in enumerate(fold_dirs):
        print(f"\n{'#' * 60}")
        print(f"  Processing {fold_dir.name} ({fold_idx + 1}/{len(fold_dirs)})...")

        groups = build_fold_dataframes(fold_dir, label_map, genome2text)

        for group_name, (train_df, val_df) in sorted(groups.items()):
            if len(train_df) == 0:
                print(f"  [ERROR] No training samples for {fold_dir.name}/{group_name} — skipping")
                continue
            if len(val_df) == 0:
                print(f"  [ERROR] No validation samples for {fold_dir.name}/{group_name} — skipping")
                continue

            metrics = finetune_fold(
                fold_dir.name, fold_idx, group_name, train_df, val_df, args, tokenizer
            )
            metrics["fold"] = fold_dir.name
            metrics["group"] = group_name
            all_metrics.append(metrics)

    if not all_metrics:
        print("\n[ERROR] No folds were successfully trained. Check your data directory.")
        return

    # ---- Aggregate summary ----
    print(f"\n{'=' * 60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Fold/Group':<30} {'Virulent':>10} {'Temperate':>11} {'Overall':>9}")
    print(f"{'-' * 63}")
    for m in all_metrics:
        key = f"{m['fold']}/{m['group']}"
        print(f"{key:<30} {m['virulent_acc']:>10.4f} "
              f"{m['temperate_acc']:>11.4f} {m['overall_acc']:>9.4f}")
    print(f"{'-' * 48}")

    viru_arr    = np.array([m["virulent_acc"]  for m in all_metrics])
    temp_arr    = np.array([m["temperate_acc"] for m in all_metrics])
    overall_arr = np.array([m["overall_acc"]   for m in all_metrics])

    print(f"{'Mean':<30} {viru_arr.mean():>10.4f} "
          f"{temp_arr.mean():>11.4f} {overall_arr.mean():>9.4f}")
    print(f"{'Std':<30} {viru_arr.std():>10.4f} "
          f"{temp_arr.std():>11.4f} {overall_arr.std():>9.4f}")

    summary = {
        "folds": all_metrics,
        "mean_virulent_acc":  float(viru_arr.mean()),
        "std_virulent_acc":   float(viru_arr.std()),
        "mean_temperate_acc": float(temp_arr.mean()),
        "std_temperate_acc":  float(temp_arr.std()),
        "mean_overall_acc":   float(overall_arr.mean()),
        "std_overall_acc":    float(overall_arr.std()),
        "label_convention":   {"0": "temperate", "1": "virulent"},
        "note": (
            "Label convention: 0=temperate, 1=virulent. "
            "This is REVERSED from the original PhaTYP model "
            "(original: 0=virulent, 1=temperate). "
            "Adjust inference accordingly."
        ),
    }

    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
