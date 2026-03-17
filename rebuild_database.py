#!/usr/bin/env python3
"""
rebuild_database.py — Extend PhaTYP protein cluster database with new phage genomes.

PIPELINE
--------
  1. Predict proteins from new genomes (Prodigal)
  2. Align new proteins against existing database (DIAMOND blastp)
       → proteins with a hit  : assigned to the matching existing PC
       → proteins without hit : sent to step 3
  3. Cluster uncovered proteins (MMseqs2 easy-cluster) → new PC IDs
  4. Update database files:
       database/database.fa.gz   (add new cluster representative sequences)
       database/proteins.csv     (add new protein → cluster mappings)
       database/pc2wordsid.dict  (add new PC → token-id mappings)
       config/vocab.txt          (append new PC tokens)
       config/config.json        (update vocab_size)
  5. (Optional) Generate pretraining corpus (train.txt / val.txt)
       from ALL genomes that now have at least one PC token

DEPENDENCIES
------------
  Python  : pip install biopython pandas
  Binaries: prodigal (or pprodigal), diamond, mmseqs2
            → conda install -c bioconda prodigal diamond mmseqs2
            → OR: apt install prodigal; conda install -c bioconda diamond mmseqs2

USAGE
-----
  # Minimal — only rebuild database
  python rebuild_database.py \\
      --new_genomes  my_4044_genomes.fasta \\
      --phatyp_dir   /path/to/PhaTYP

  # Also generate pretraining corpus from ALL labeled genomes
  python rebuild_database.py \\
      --new_genomes  my_4044_genomes.fasta \\
      --phatyp_dir   /path/to/PhaTYP \\
      --all_genomes  all_genomes_for_pretrain.fasta \\
      --generate_pretrain_corpus \\
      --output_pretrain_dir  pretrain_data
"""

import os
import sys
import csv
import gzip
import json
import math
import pickle
import random
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_tool(name):
    """Return path to tool or exit with an error message."""
    path = shutil.which(name)
    if path is None:
        sys.exit(
            f"\n[ERROR] '{name}' not found in PATH.\n"
            f"  Install with:  conda install -c bioconda {name}\n"
        )
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extend PhaTYP database with new phage genomes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--new_genomes", required=True,
        help="FASTA file of new phage genomes (DNA/nucleotide sequences)",
    )
    parser.add_argument(
        "--phatyp_dir", default=".",
        help="Root directory of PhaTYP (contains database/ and config/)",
    )
    parser.add_argument(
        "--workdir", default="rebuild_workdir",
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of CPU threads for Prodigal / DIAMOND / MMseqs2",
    )

    # DIAMOND alignment parameters
    parser.add_argument(
        "--diamond_evalue", type=float, default=1e-5,
        help="E-value threshold for DIAMOND blastp (proteins below this "
             "threshold are considered 'covered' by an existing PC)",
    )
    parser.add_argument(
        "--diamond_identity", type=float, default=0.0,
        help="Minimum sequence identity for DIAMOND hits (0 = no filter)",
    )

    # MMseqs2 clustering parameters for NEW (uncovered) proteins
    parser.add_argument(
        "--mmseqs_min_seq_id", type=float, default=0.3,
        help="MMseqs2 --min-seq-id for clustering new proteins into new PCs",
    )
    parser.add_argument(
        "--mmseqs_cov", type=float, default=0.8,
        help="MMseqs2 -c (coverage) for clustering new proteins",
    )
    parser.add_argument(
        "--mmseqs_cov_mode", type=int, default=0,
        help="MMseqs2 --cov-mode (0=bidirectional, 1=query, 2=target)",
    )

    # Pretraining corpus generation (optional)
    parser.add_argument(
        "--generate_pretrain_corpus", action="store_true",
        help="After updating the database, generate pretraining corpus "
             "(train.txt + val.txt) from --all_genomes",
    )
    parser.add_argument(
        "--all_genomes",
        help="FASTA file of ALL genomes to use for pretraining corpus "
             "(should include both existing and new genomes). "
             "Required when --generate_pretrain_corpus is set.",
    )
    parser.add_argument(
        "--output_pretrain_dir", default="pretrain_data",
        help="Directory to write train.txt and val.txt",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of genomes to use as validation set (0–1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--min_len", type=int, default=0,
        help="Minimum genome length (bp) to keep (0 = keep all)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – Prodigal: DNA → protein FASTA
# ---------------------------------------------------------------------------

def run_prodigal(genome_fasta, output_faa, threads, min_len=0):
    """Run Prodigal (or pprodigal) to predict proteins from genome_fasta."""
    print("\n[Step 1] Predicting proteins with Prodigal...")

    # Filter by length if requested
    from Bio import SeqIO
    records = [r for r in SeqIO.parse(genome_fasta, "fasta")
               if len(r.seq) >= min_len]
    print(f"  Genomes passing length filter (>={min_len} bp): {len(records)}")

    if len(records) == 0:
        sys.exit(f"[ERROR] No genomes passed the length filter ({min_len} bp).")

    # Write filtered genomes to temp file
    filtered_fasta = output_faa.replace(".faa", "_filtered_genomes.fa")
    SeqIO.write(records, filtered_fasta, "fasta")

    # Choose prodigal binary (parallelized version if available)
    prodigal_bin = shutil.which("pprodigal")
    if prodigal_bin:
        cmd = f"pprodigal -T {threads} -i {filtered_fasta} -a {output_faa} -f gff -p meta"
        print(f"  Using parallelized pprodigal (threads={threads})")
    else:
        prodigal_bin = check_tool("prodigal")
        cmd = f"prodigal -i {filtered_fasta} -a {output_faa} -f gff -p meta"
        print("  Using standard prodigal (single thread)")

    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Count predicted proteins
    from Bio import SeqIO as _SeqIO
    n_proteins = sum(1 for _ in _SeqIO.parse(output_faa, "fasta"))
    print(f"  Predicted proteins: {n_proteins}")
    return n_proteins


# ---------------------------------------------------------------------------
# Step 2 – DIAMOND: map new proteins to existing PCs
# ---------------------------------------------------------------------------

def run_diamond(new_proteins_faa, db_fasta_gz, workdir, evalue, identity, threads):
    """
    Align new proteins against existing database using DIAMOND blastp.
    Returns a dict: protein_id → existing_pc_name  (only covered proteins)
    """
    print("\n[Step 2] Aligning new proteins against existing database (DIAMOND)...")
    check_tool("diamond")

    dmnd_db = os.path.join(workdir, "existing_db.dmnd")
    results_tab = os.path.join(workdir, "diamond_results.tab")

    # Build DIAMOND database from the gzipped FASTA
    print("  Building DIAMOND database...")
    make_db_cmd = (
        f"diamond makedb --threads {threads} "
        f"--in {db_fasta_gz} "
        f"-d {dmnd_db}"
    )
    subprocess.run(make_db_cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run DIAMOND blastp — keep only top-1 hit per query
    identity_flag = f"--id {identity * 100:.1f}" if identity > 0 else ""
    blastp_cmd = (
        f"diamond blastp --threads {threads} --sensitive "
        f"-d {dmnd_db} "
        f"-q {new_proteins_faa} "
        f"-o {results_tab} "
        f"-k 1 "
        f"-e {evalue} "
        f"{identity_flag} "
        f"--outfmt 6 qseqid sseqid pident evalue"
    )
    print(f"  Running DIAMOND blastp (e-value ≤ {evalue})...")
    subprocess.run(blastp_cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load existing proteins.csv to map hit_protein → PC
    proteins_csv = os.path.join(workdir, "_proteins.csv")   # copied in main()
    protein2pc = {}
    with open(proteins_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["cluster"]:
                protein2pc[row["protein_id"]] = row["cluster"]

    # Parse DIAMOND results
    covered = {}   # new_protein_id → pc_name
    with open(results_tab) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            query, ref = parts[0], parts[1]
            if ref in protein2pc:
                covered[query] = protein2pc[ref]

    print(f"  Covered by existing PCs : {len(covered)}")
    return covered


# ---------------------------------------------------------------------------
# Step 3 – MMseqs2: cluster uncovered proteins → new PCs
# ---------------------------------------------------------------------------

def run_mmseqs_cluster(uncovered_faa, workdir, min_seq_id, cov, cov_mode, threads,
                       next_pc_index):
    """
    Cluster uncovered proteins with MMseqs2.
    Returns:
        new_protein2pc  : dict  protein_id → new PC name
        representative_records : list of SeqRecord (one per new cluster)
        n_new_pcs : int
    """
    print("\n[Step 3] Clustering uncovered proteins with MMseqs2...")
    check_tool("mmseqs")

    from Bio import SeqIO

    n_uncovered = sum(1 for _ in SeqIO.parse(uncovered_faa, "fasta"))
    print(f"  Uncovered proteins to cluster: {n_uncovered}")

    if n_uncovered == 0:
        print("  No uncovered proteins — skipping MMseqs2 step.")
        return {}, [], 0

    mmseqs_outdir = os.path.join(workdir, "mmseqs_out")
    mmseqs_tmpdir = os.path.join(workdir, "mmseqs_tmp")
    os.makedirs(mmseqs_outdir, exist_ok=True)
    os.makedirs(mmseqs_tmpdir, exist_ok=True)

    cluster_prefix = os.path.join(mmseqs_outdir, "new_clusters")

    # Run easy-cluster
    cmd = (
        f"mmseqs easy-cluster "
        f"{uncovered_faa} "
        f"{cluster_prefix} "
        f"{mmseqs_tmpdir} "
        f"--threads {threads} "
        f"--min-seq-id {min_seq_id} "
        f"-c {cov} "
        f"--cov-mode {cov_mode}"
    )
    print(f"  Running: mmseqs easy-cluster (min-seq-id={min_seq_id}, cov={cov})")
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Parse cluster TSV  (representative \t member)
    cluster_tsv = cluster_prefix + "_cluster.tsv"
    rep2members = {}   # representative → [member1, member2, ...]
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            rep2members.setdefault(rep, []).append(member)

    # Load representative sequences
    rep_faa = cluster_prefix + "_rep_seq.fasta"
    rep_seq = {r.id: r for r in SeqIO.parse(rep_faa, "fasta")}

    # Assign PC names (PC_045578, PC_045579, ...)
    new_protein2pc = {}
    representative_records = []
    n_new_pcs = 0

    for rep, members in rep2members.items():
        pc_name = f"PC_{next_pc_index + n_new_pcs:06d}"
        n_new_pcs += 1
        for member in members:
            new_protein2pc[member] = pc_name
        # Keep representative sequence, rename header to pc_name for clarity
        if rep in rep_seq:
            rec = rep_seq[rep]
            rec.id = rep          # keep original ID (protein accession)
            rec.description = f"representative of {pc_name}"
            representative_records.append((pc_name, rec))

    print(f"  New protein clusters (new PCs): {n_new_pcs}")
    return new_protein2pc, representative_records, n_new_pcs


# ---------------------------------------------------------------------------
# Step 4 – Update database files
# ---------------------------------------------------------------------------

def update_database(phatyp_dir, workdir, new_proteins_faa,
                    covered_protein2pc, new_protein2pc, representative_records):
    """
    Update all four database/config files in-place (with backups).
    Returns (n_new_pcs, updated_total_vocab_size).
    """
    print("\n[Step 4] Updating database and config files...")

    db_dir     = os.path.join(phatyp_dir, "database")
    config_dir = os.path.join(phatyp_dir, "config")

    # --- 4a. proteins.csv ---
    proteins_csv_path = os.path.join(db_dir, "proteins.csv")
    _backup(proteins_csv_path)

    from Bio import SeqIO

    # Map new protein IDs → contig IDs (from Prodigal header: contig_proteinNum)
    protein2contig = {}
    for rec in SeqIO.parse(new_proteins_faa, "fasta"):
        # Prodigal format: ">contigID_N ..."  → contigID = rsplit("_", 1)[0]
        contig = rec.id.rsplit("_", 1)[0]
        protein2contig[rec.id] = contig

    all_new_protein2pc = {**covered_protein2pc, **new_protein2pc}

    new_rows = []
    for protein_id, pc in all_new_protein2pc.items():
        contig_id = protein2contig.get(protein_id, "")
        new_rows.append(f"{protein_id},{contig_id},hypothetical protein,{pc}\n")

    print(f"  Appending {len(new_rows)} new protein entries to proteins.csv")
    with open(proteins_csv_path, "a") as f:
        f.writelines(new_rows)

    # --- 4b. database.fa.gz --- (add new representative sequences)
    db_fasta_gz_path = os.path.join(db_dir, "database.fa.gz")
    _backup(db_fasta_gz_path)

    from Bio import SeqIO as _SeqIO
    import io

    n_new_reps = len(representative_records)
    print(f"  Adding {n_new_reps} representative sequences to database.fa.gz")
    with gzip.open(db_fasta_gz_path, "at") as gz_f:
        for pc_name, rec in representative_records:
            gz_f.write(f">{rec.id} {rec.description}\n")
            seq = str(rec.seq)
            # Write in 60-char lines
            for i in range(0, len(seq), 60):
                gz_f.write(seq[i:i+60] + "\n")

    # --- 4c. pc2wordsid.dict ---
    pc2wordsid_path = os.path.join(db_dir, "pc2wordsid.dict")
    _backup(pc2wordsid_path)

    with open(pc2wordsid_path, "rb") as f:
        pc2wordsid = pickle.load(f)

    existing_max_id = max(pc2wordsid.values()) if pc2wordsid else -1
    new_pcs = sorted(set(new_protein2pc.values()))
    next_word_id = existing_max_id + 1

    for pc_name in new_pcs:
        if pc_name not in pc2wordsid:
            pc2wordsid[pc_name] = next_word_id
            next_word_id += 1

    with open(pc2wordsid_path, "wb") as f:
        pickle.dump(pc2wordsid, f)

    print(f"  pc2wordsid.dict: added {len(new_pcs)} new PC → token-id entries "
          f"(total PCs: {len(pc2wordsid)})")

    # --- 4d. config/vocab.txt ---
    vocab_path = os.path.join(config_dir, "vocab.txt")
    _backup(vocab_path)

    print(f"  Appending {len(new_pcs)} new PC tokens to vocab.txt")
    with open(vocab_path, "a") as f:
        for pc_name in new_pcs:
            f.write(pc_name + "\n")

    new_vocab_size = sum(1 for _ in open(vocab_path))
    print(f"  New vocab size: {new_vocab_size}")

    # --- 4e. config/config.json ---
    config_json_path = os.path.join(config_dir, "config.json")
    _backup(config_json_path)

    with open(config_json_path) as f:
        config = json.load(f)
    old_vocab_size = config.get("vocab_size", "?")
    config["vocab_size"] = new_vocab_size
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json vocab_size: {old_vocab_size} → {new_vocab_size}")

    return len(new_pcs), new_vocab_size


def _backup(path):
    """Create a .bak copy of a file before modifying it."""
    bak = path + ".bak"
    if not os.path.exists(bak):   # only backup once (first run)
        shutil.copy2(path, bak)
        print(f"  Backup: {os.path.basename(path)} → {os.path.basename(bak)}")


# ---------------------------------------------------------------------------
# Step 5 – Generate pretraining corpus
# ---------------------------------------------------------------------------

def generate_pretrain_corpus(phatyp_dir, all_genomes_fasta, output_dir,
                             val_split, seed, threads, min_len, workdir):
    """
    Run preprocessing.py on ALL genomes (with updated database) and
    convert bert_feat.csv into train.txt + val.txt for pretrain.py.
    """
    print("\n[Step 5] Generating pretraining corpus...")
    os.makedirs(output_dir, exist_ok=True)

    # Run preprocessing.py
    preprocess_script = os.path.join(phatyp_dir, "preprocessing.py")
    preprocess_mid = os.path.join(workdir, "pretrain_preprocessing")
    os.makedirs(preprocess_mid, exist_ok=True)

    cmd = [
        sys.executable, preprocess_script,
        "--contigs", all_genomes_fasta,
        "--midfolder", preprocess_mid,
        "--len", str(min_len),
        "--threads", str(threads),
    ]
    print(f"  Running preprocessing.py (this may take a while)...")
    log_path = os.path.join(workdir, "pretrain_preprocessing.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, cwd=phatyp_dir, stdout=log_f, stderr=log_f)
    if result.returncode != 0:
        print(f"  [WARN] preprocessing.py exited with code {result.returncode}. "
              f"Check {log_path}")
    print(f"  Preprocessing log: {log_path}")

    # Load bert_feat.csv
    import csv as _csv
    bert_feat_path = os.path.join(preprocess_mid, "bert_feat.csv")
    if not os.path.exists(bert_feat_path):
        sys.exit(f"[ERROR] bert_feat.csv not found at {bert_feat_path}. "
                 f"Preprocessing may have failed.")

    sentences = []
    with open(bert_feat_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                sentences.append(text)

    print(f"  Genomes with PC features: {len(sentences)}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(sentences)
    n_val   = max(1, int(len(sentences) * val_split))
    n_train = len(sentences) - n_val

    train_sentences = sentences[:n_train]
    val_sentences   = sentences[n_train:]

    train_path = os.path.join(output_dir, "train.txt")
    val_path   = os.path.join(output_dir, "val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_sentences) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_sentences) + "\n")

    print(f"  train.txt: {n_train} genomes  →  {train_path}")
    print(f"  val.txt  : {n_val}   genomes  →  {val_path}")
    print(f"\n  To start pretraining, run:")
    print(f"    cd {phatyp_dir}/train")
    print(f"    python pretrain.py --train {os.path.abspath(train_path)} "
          f"--val {os.path.abspath(val_path)}")
    print(f"  NOTE: Update NUM_TOKEN in pretrain.py to match the new vocab size.")

    return n_train, n_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    phatyp_dir = os.path.abspath(args.phatyp_dir)
    workdir    = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    print("=" * 65)
    print("  PhaTYP Database Rebuild")
    print("=" * 65)
    print(f"  PhaTYP dir  : {phatyp_dir}")
    print(f"  New genomes : {args.new_genomes}")
    print(f"  Work dir    : {workdir}")

    # Validate paths
    for path, name in [
        (os.path.join(phatyp_dir, "database", "database.fa.gz"), "database.fa.gz"),
        (os.path.join(phatyp_dir, "database", "proteins.csv"),   "proteins.csv"),
        (os.path.join(phatyp_dir, "database", "pc2wordsid.dict"),"pc2wordsid.dict"),
        (os.path.join(phatyp_dir, "config",   "vocab.txt"),      "vocab.txt"),
        (os.path.join(phatyp_dir, "config",   "config.json"),    "config.json"),
    ]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Required file not found: {path}")

    if args.generate_pretrain_corpus and not args.all_genomes:
        sys.exit("[ERROR] --generate_pretrain_corpus requires --all_genomes")

    # Copy proteins.csv to workdir so DIAMOND step can reference it
    shutil.copy2(
        os.path.join(phatyp_dir, "database", "proteins.csv"),
        os.path.join(workdir, "_proteins.csv"),
    )

    # ------------------------------------------------------------------ #
    # Step 1 – Predict proteins                                           #
    # ------------------------------------------------------------------ #
    new_proteins_faa = os.path.join(workdir, "new_proteins.faa")
    run_prodigal(
        genome_fasta=args.new_genomes,
        output_faa=new_proteins_faa,
        threads=args.threads,
        min_len=args.min_len,
    )

    # ------------------------------------------------------------------ #
    # Step 2 – DIAMOND blastp vs existing database                        #
    # ------------------------------------------------------------------ #
    covered_protein2pc = run_diamond(
        new_proteins_faa=new_proteins_faa,
        db_fasta_gz=os.path.join(phatyp_dir, "database", "database.fa.gz"),
        workdir=workdir,
        evalue=args.diamond_evalue,
        identity=args.diamond_identity,
        threads=args.threads,
    )

    # ------------------------------------------------------------------ #
    # Step 3 – Cluster uncovered proteins with MMseqs2                    #
    # ------------------------------------------------------------------ #
    # Collect uncovered proteins
    from Bio import SeqIO
    uncovered_faa = os.path.join(workdir, "uncovered_proteins.faa")
    uncovered_records = [
        rec for rec in SeqIO.parse(new_proteins_faa, "fasta")
        if rec.id not in covered_protein2pc
    ]
    SeqIO.write(uncovered_records, uncovered_faa, "fasta")
    print(f"\n  Proteins NOT covered by existing PCs: {len(uncovered_records)}")

    # Determine next PC index
    with open(os.path.join(phatyp_dir, "database", "pc2wordsid.dict"), "rb") as f:
        existing_pc2wordsid = pickle.load(f)
    # PC names are PC_XXXXXX; find the max numeric index
    max_pc_idx = max(
        (int(pc.split("_")[1]) for pc in existing_pc2wordsid.keys() if "_" in pc),
        default=-1,
    )
    next_pc_index = max_pc_idx + 1

    new_protein2pc, representative_records, n_new_pcs = run_mmseqs_cluster(
        uncovered_faa=uncovered_faa,
        workdir=workdir,
        min_seq_id=args.mmseqs_min_seq_id,
        cov=args.mmseqs_cov,
        cov_mode=args.mmseqs_cov_mode,
        threads=args.threads,
        next_pc_index=next_pc_index,
    )

    # ------------------------------------------------------------------ #
    # Step 4 – Update database files                                      #
    # ------------------------------------------------------------------ #
    n_new_pcs_added, new_vocab_size = update_database(
        phatyp_dir=phatyp_dir,
        workdir=workdir,
        new_proteins_faa=new_proteins_faa,
        covered_protein2pc=covered_protein2pc,
        new_protein2pc=new_protein2pc,
        representative_records=representative_records,
    )

    # ------------------------------------------------------------------ #
    # Step 5 – (Optional) Generate pretraining corpus                     #
    # ------------------------------------------------------------------ #
    if args.generate_pretrain_corpus:
        generate_pretrain_corpus(
            phatyp_dir=phatyp_dir,
            all_genomes_fasta=args.all_genomes,
            output_dir=os.path.abspath(args.output_pretrain_dir),
            val_split=args.val_split,
            seed=args.seed,
            threads=args.threads,
            min_len=args.min_len,
            workdir=workdir,
        )

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Proteins covered by existing PCs : {len(covered_protein2pc)}")
    print(f"  New PCs created (from MMseqs2)   : {n_new_pcs_added}")
    print(f"  New vocab size                   : {new_vocab_size}")
    print()
    print("  Updated files (originals backed up as *.bak):")
    print(f"    database/proteins.csv")
    print(f"    database/database.fa.gz")
    print(f"    database/pc2wordsid.dict")
    print(f"    config/vocab.txt")
    print(f"    config/config.json")
    print()
    if n_new_pcs_added > 0:
        print("  NEXT STEPS:")
        print(f"  1. Update NUM_TOKEN in train/pretrain.py:")
        print(f"       NUM_TOKEN = {new_vocab_size}  # was 45583")
        print(f"  2. Run pretrain.py with the new corpus:")
        print(f"       cd train/")
        print(f"       python pretrain.py --train ../pretrain_data/train.txt "
              f"--val ../pretrain_data/val.txt")
        print(f"  3. After pretraining, run kfold fine-tuning as usual.")
    else:
        print("  No new PCs were added — database is already sufficient.")
        print("  You can re-run fine-tuning directly.")
    print("=" * 65)


if __name__ == "__main__":
    main()
