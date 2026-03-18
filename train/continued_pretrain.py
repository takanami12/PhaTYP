"""
continued_pretrain.py — Domain-Adaptive PreTraining (DAPT) for PhaTYP.

Load the original PhaTYP pretrained model, resize the token embedding
layer to cover newly added PC tokens, then run a short MLM training on
the new genomes so that new PC embeddings are learned before fine-tuning.

Usage (from PhaTYP/train/ directory):
    python continued_pretrain.py --train train.txt --val val.txt

After this script completes, run fine-tuning with:
    python finetune.py --train train.csv --val val.csv
  or:
    cd ..
    python run_kfold_finetune.py --pretrained_model train/log_dapt ...
"""

import argparse

import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class LineByLineDataset(Dataset):
    """Replacement for the removed LineByLineTextDataset."""
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.examples = tokenizer(
            lines,
            max_length=block_size,
            truncation=True,
            padding=False,
        )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.examples[i], dtype=torch.long)}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_MODEL_DIR = "/content/PhaTYP/model"   # original PhaTYP pretrained model
CONFIG_DIR         = "../config"               # tokenizer / updated vocab.txt
OUTPUT_DIR         = "log_dapt"               # where the DAPT model is saved

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Domain-Adaptive PreTraining (DAPT) for PhaTYP with new genomes."
)
parser.add_argument("--train", type=str, default="train.txt",
                    help="Plain-text pretraining corpus (one genome sentence per line)")
parser.add_argument("--val",   type=str, default="val.txt",
                    help="Plain-text validation corpus")
parser.add_argument("--max_steps", type=int, default=20000,
                    help="MLM training steps (default 20000; much less than full pretrain)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate (lower than full pretrain to preserve existing weights)")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=2)
inputs = parser.parse_args()

SENTENCE_LEN = 300

# ---------------------------------------------------------------------------
# Tokenizer with updated vocab (includes new PCs added by rebuild_database.py)
# ---------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(CONFIG_DIR, do_basic_tokenize=False)
new_vocab_size = len(tokenizer)
print(f"Vocab size (updated): {new_vocab_size}")

# ---------------------------------------------------------------------------
# Load original pretrained model and expand embedding layer
# ---------------------------------------------------------------------------
print(f"Loading original model from: {ORIGINAL_MODEL_DIR}")
model = BertForMaskedLM.from_pretrained(
    ORIGINAL_MODEL_DIR,
    ignore_mismatched_sizes=True,   # handles fine-tuned → MLM head mismatch
)

old_vocab_size = model.config.vocab_size
if old_vocab_size != new_vocab_size:
    print(f"Resizing token embeddings: {old_vocab_size} → {new_vocab_size}")
    model.resize_token_embeddings(new_vocab_size)
else:
    print("Vocab size unchanged — no embedding resize needed.")

print(f"Model parameters: {model.num_parameters():,}")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
data_train = LineByLineDataset(tokenizer, inputs.train, SENTENCE_LEN)
data_val   = LineByLineDataset(tokenizer, inputs.val,   SENTENCE_LEN)
print(f"Train sentences: {len(data_train)} | Val sentences: {len(data_val)}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.025,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    max_steps=inputs.max_steps,
    per_device_train_batch_size=inputs.batch_size,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=inputs.grad_accum,
    evaluation_strategy="epoch",
    learning_rate=inputs.lr,
    weight_decay=0.01,
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.98,
    warmup_steps=1000,
    fp16=True,
    save_steps=2000,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data_train,
    eval_dataset=data_val,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"\nDAPT model saved to: {OUTPUT_DIR}")
print("Next step: run finetune.py or run_kfold_finetune.py")
