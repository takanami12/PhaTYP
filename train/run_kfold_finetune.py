#!/usr/bin/env python3
"""
K-Fold Fine-tuning for PhaTYP model.

PIPELINE:
  1. Collect all unique genomes across all folds/groups
  2. Run preprocessing ONCE (Prodigal + DIAMOND) -> PC token features
  3. For each fold: assemble train/val CSVs from all groups, fine-tune, evaluate

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
      --pretrained_model train/log \
      --output_dir kfold_results

  # Multi-GPU (e.g., 4 GPUs):
  torchrun --nproc_per_node=4 train/run_kfold_finetune.py \
      --data_dir  /path/to/data \
      --label_csv /path/to/labels.csv \
      --pretrained_model train/log \
      --output_dir kfold_results
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import datasets
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from Bio import SeqIO


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
        "--pretrained_model", default="train/log",
        help="Pretrained PhaTYP model directory (output of pretrain.py). "
             "Must be run from PhaTYP root directory.",
    )
    parser.add_argument(
        "--config_dir", default="config",
        help="BERT config/tokenizer directory (PhaTYP root/config)",
    )
    parser.add_argument(
        "--output_dir", default="kfold_results",
        help="Output directory for models, logs, and results",
    )
    parser.add_argument(
        "--phatyp_dir", default=".",
        help="PhaTYP root directory (contains preprocessing.py and database/)",
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
        "--min_len", type=int, default=100,
        help="Minimum contig length (bp) to keep during preprocessing",
    )
    parser.add_argument(
        "--threads", type=str, default="8",
        help="Number of threads for DIAMOND and Prodigal",
    )
    parser.add_argument(
        "--skip_preprocessing", action="store_true",
        help="Skip preprocessing step and load cached features from output_dir/preprocessing/",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended for A100/H100)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")
    return parser.parse_args()


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
    """Return list of sequence IDs from a FASTA file."""
    return [r.id for r in SeqIO.parse(str(fasta_path), "fasta")]


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
    subprocess.run(cmd, check=True, cwd=phatyp_dir)

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
        genome_id = id2contig[idx]
        genome2text[genome_id] = text

    print(f"  Loaded features for {len(genome2text)} genomes")
    return genome2text


# ---------------------------------------------------------------------------
# Step 2 – Build per-fold train/val DataFrames
# ---------------------------------------------------------------------------

def build_fold_dataframes(fold_dir: Path, label_map: dict, genome2text: dict):
    """
    Build train and val DataFrames for one fold by merging all groups.

    Returns:
        train_df, val_df  (columns: label, text)
    """
    train_records, val_records = [], []
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

        for genome_id in get_genome_ids(fasta_path):
            if genome_id not in genome2text:
                missing_feat.append(genome_id)
                continue
            if genome_id not in label_map:
                missing_label.append(genome_id)
                continue
            record = {"label": label_map[genome_id], "text": genome2text[genome_id]}
            if split == "train":
                train_records.append(record)
            else:
                val_records.append(record)

    if missing_feat:
        print(f"  [WARN] {len(missing_feat)} genomes skipped (no features — likely too short)")
    if missing_label:
        print(f"  [WARN] {len(missing_label)} genomes skipped (no label in label CSV)")

    return pd.DataFrame(train_records), pd.DataFrame(val_records)


# ---------------------------------------------------------------------------
# Step 3 – Fine-tune + evaluate one fold
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Compute classification metrics. Label convention: 0=temperate, 1=virulent.
    Returns accuracy, F1, MCC, AUC, sensitivity, specificity for thesis comparison.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # virulent recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # temperate recall
    overall_acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    f1 = f1_score(labels, preds, zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    try:
        from scipy.special import softmax as sp_softmax
        probs = sp_softmax(logits, axis=-1)[:, 1]
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = -1.0

    return {
        "overall_acc":   round(overall_acc, 4),
        "sensitivity":   round(sensitivity, 4),   # virulent recall
        "specificity":   round(specificity, 4),   # temperate recall
        "f1":            round(f1, 4),
        "f1_macro":      round(f1_macro, 4),
        "mcc":           round(mcc, 4),
        "auc":           round(auc, 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def finetune_fold(fold_name, fold_idx, train_df, val_df, args, tokenizer):
    """Fine-tune model for one fold and return evaluation metrics."""
    fold_out = os.path.join(args.output_dir, fold_name)
    os.makedirs(fold_out, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx} ({fold_name})")
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
        overwrite_output_dir=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        logging_dir=os.path.join(fold_out, "logs"),
        logging_steps=50,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_data["train"],
        eval_dataset=tok_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final prediction on val set
    pred_out = trainer.predict(tok_data["test"])
    metrics  = compute_metrics((pred_out.predictions, pred_out.label_ids))

    print(f"\n  [Fold {fold_idx}] acc={metrics['overall_acc']:.4f} "
          f"| sensitivity={metrics['sensitivity']:.4f} | specificity={metrics['specificity']:.4f} "
          f"| f1={metrics['f1']:.4f} | mcc={metrics['mcc']:.4f} | auc={metrics['auc']:.4f}")
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

    # Enable TF32 on Ampere+ GPUs (A100/H100) for faster fp32 math
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

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
    print(f"\n[4] Starting K-Fold fine-tuning ({len(fold_dirs)} folds)...")
    all_metrics = []

    for fold_idx, fold_dir in enumerate(fold_dirs):
        print(f"\n{'#' * 60}")
        print(f"  Processing {fold_dir.name} ({fold_idx + 1}/{len(fold_dirs)})...")

        train_df, val_df = build_fold_dataframes(fold_dir, label_map, genome2text)

        if len(train_df) == 0:
            print(f"  [ERROR] No training samples for {fold_dir.name} — skipping fold")
            continue
        if len(val_df) == 0:
            print(f"  [ERROR] No validation samples for {fold_dir.name} — skipping fold")
            continue

        metrics = finetune_fold(
            fold_dir.name, fold_idx, train_df, val_df, args, tokenizer
        )
        metrics["fold"] = fold_dir.name
        all_metrics.append(metrics)

    if not all_metrics:
        print("\n[ERROR] No folds were successfully trained. Check your data directory.")
        return

    # ---- Aggregate summary ----
    print(f"\n{'=' * 70}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Fold':<15} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'MCC':>7} {'AUC':>7}")
    print(f"{'-' * 70}")
    for m in all_metrics:
        print(f"{m['fold']:<15} {m['overall_acc']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m['specificity']:>7.4f} {m['f1']:>7.4f} {m['mcc']:>7.4f} {m['auc']:>7.4f}")
    print(f"{'-' * 70}")

    def _arr(key): return np.array([m[key] for m in all_metrics])

    acc_arr  = _arr("overall_acc")
    sens_arr = _arr("sensitivity")
    spec_arr = _arr("specificity")
    f1_arr   = _arr("f1")
    mcc_arr  = _arr("mcc")
    auc_arr  = _arr("auc")

    print(f"{'Mean':<15} {acc_arr.mean():>7.4f} {sens_arr.mean():>7.4f} "
          f"{spec_arr.mean():>7.4f} {f1_arr.mean():>7.4f} {mcc_arr.mean():>7.4f} {auc_arr.mean():>7.4f}")
    print(f"{'Std':<15} {acc_arr.std():>7.4f} {sens_arr.std():>7.4f} "
          f"{spec_arr.std():>7.4f} {f1_arr.std():>7.4f} {mcc_arr.std():>7.4f} {auc_arr.std():>7.4f}")

    summary = {
        "folds": all_metrics,
        "mean_overall_acc":   float(acc_arr.mean()),  "std_overall_acc":   float(acc_arr.std()),
        "mean_sensitivity":   float(sens_arr.mean()), "std_sensitivity":   float(sens_arr.std()),
        "mean_specificity":   float(spec_arr.mean()), "std_specificity":   float(spec_arr.std()),
        "mean_f1":            float(f1_arr.mean()),   "std_f1":            float(f1_arr.std()),
        "mean_mcc":           float(mcc_arr.mean()),  "std_mcc":           float(mcc_arr.std()),
        "mean_auc":           float(auc_arr.mean()),  "std_auc":           float(auc_arr.std()),
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
