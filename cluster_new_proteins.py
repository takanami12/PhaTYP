#!/usr/bin/env python
"""
Script để cluster proteins mới vào các protein clusters hiện có của PhaTYP.

Sử dụng DIAMOND để align proteins mới với database hiện tại, sau đó gán cluster
dựa trên best hit.

Cách sử dụng:
    python cluster_new_proteins.py --input new_proteins.fa --output new_proteins_clustered.csv

Output: File CSV với format phù hợp để dùng với update_database.py
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from Bio import SeqIO
import sys


class ProteinClusterer:
    def __init__(self, database_dir="database", threads=8):
        self.database_dir = Path(database_dir)
        self.threads = threads

        # Database files
        self.database_fa_gz = self.database_dir / "database.fa.gz"
        self.proteins_csv = self.database_dir / "proteins.csv"

        # Check if database exists
        if not self.database_fa_gz.exists():
            raise FileNotFoundError(f"Database không tồn tại: {self.database_fa_gz}")

        if not self.proteins_csv.exists():
            raise FileNotFoundError(f"File proteins.csv không tồn tại: {self.proteins_csv}")

        print("✓ Database files found")

    def check_dependencies(self):
        """Kiểm tra xem DIAMOND có được cài đặt không"""
        try:
            result = subprocess.run(
                ["diamond", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.split('\n')[0]
            print(f"✓ DIAMOND found: {version}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ DIAMOND không được tìm thấy!")
            print("💡 Cài đặt: conda install -c bioconda diamond")
            return False

    def prepare_database(self, temp_dir):
        """Giải nén và tạo DIAMOND database"""
        print("\n📂 Đang chuẩn bị DIAMOND database...")

        # Decompress database
        db_fa = temp_dir / "database.fa"
        with subprocess.Popen(
            ["gunzip", "-c", str(self.database_fa_gz)],
            stdout=subprocess.PIPE
        ) as gunzip_proc:
            with open(db_fa, 'wb') as f_out:
                shutil.copyfileobj(gunzip_proc.stdout, f_out)

        print("✓ Database đã giải nén")

        # Create DIAMOND database
        db_dmnd = temp_dir / "database.dmnd"
        cmd = [
            "diamond", "makedb",
            "--in", str(db_fa),
            "--db", str(db_dmnd),
            "--quiet"
        ]

        subprocess.run(cmd, check=True)
        print("✓ DIAMOND database đã tạo")

        return db_dmnd

    def run_diamond_blastp(self, query_fa, db_dmnd, temp_dir):
        """Chạy DIAMOND BLASTP alignment"""
        print(f"\n🔍 Đang align proteins với database (sử dụng {self.threads} threads)...")

        output_file = temp_dir / "blastp_results.txt"

        cmd = [
            "diamond", "blastp",
            "--query", str(query_fa),
            "--db", str(db_dmnd),
            "--out", str(output_file),
            "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "evalue", "bitscore",
            "--max-target-seqs", "1",  # Only get best hit
            "--evalue", "1e-5",
            "--threads", str(self.threads),
            "--quiet"
        ]

        subprocess.run(cmd, check=True)
        print("✓ DIAMOND alignment hoàn tất")

        return output_file

    def parse_diamond_results(self, diamond_output):
        """Parse DIAMOND results"""
        print("\n📊 Đang parse alignment results...")

        df = pd.read_csv(
            diamond_output,
            sep='\t',
            header=None,
            names=['query_id', 'subject_id', 'identity', 'length', 'evalue', 'bitscore']
        )

        print(f"✓ Tìm thấy {len(df)} alignments")
        return df

    def assign_clusters(self, diamond_df, query_fasta):
        """Gán protein clusters dựa trên best hits"""
        print("\n🔗 Đang gán protein clusters...")

        # Load protein -> cluster mapping
        proteins_df = pd.read_csv(self.proteins_csv)
        protein_to_cluster = dict(zip(proteins_df['protein_id'], proteins_df['cluster']))

        # Parse query FASTA to get descriptions
        protein_info = {}
        for record in SeqIO.parse(query_fasta, "fasta"):
            protein_info[record.id] = {
                'description': record.description.replace(record.id, '').strip(),
                'contig_id': record.id.rsplit('_', 1)[0] if '_' in record.id else 'unknown'
            }

        # Assign clusters
        results = []
        hits_with_cluster = 0
        hits_without_cluster = 0
        no_hits = 0

        all_query_ids = set(protein_info.keys())
        query_ids_with_hits = set(diamond_df['query_id'])
        query_ids_no_hits = all_query_ids - query_ids_with_hits

        # Process proteins with hits
        for _, row in diamond_df.iterrows():
            query_id = row['query_id']
            subject_id = row['subject_id']

            # Get cluster from subject protein
            cluster = protein_to_cluster.get(subject_id, '')

            info = protein_info[query_id]
            results.append({
                'protein_id': query_id,
                'contig_id': info['contig_id'],
                'keywords': info['description'] or 'hypothetical protein',
                'cluster': cluster if pd.notna(cluster) and cluster != '' else '',
                'best_hit': subject_id,
                'identity': row['identity'],
                'evalue': row['evalue']
            })

            if cluster and cluster != '':
                hits_with_cluster += 1
            else:
                hits_without_cluster += 1

        # Process proteins without hits
        for query_id in query_ids_no_hits:
            info = protein_info[query_id]
            results.append({
                'protein_id': query_id,
                'contig_id': info['contig_id'],
                'keywords': info['description'] or 'hypothetical protein',
                'cluster': '',
                'best_hit': '',
                'identity': 0,
                'evalue': float('inf')
            })
            no_hits += 1

        results_df = pd.DataFrame(results)

        print(f"✓ Proteins với cluster: {hits_with_cluster}")
        print(f"⚠️  Proteins không có cluster (best hit không có cluster): {hits_without_cluster}")
        print(f"⚠️  Proteins không có hit: {no_hits}")

        return results_df

    def cluster(self, input_fasta, output_csv):
        """Main clustering function"""
        print("=" * 60)
        print("🚀 BẮT ĐẦU CLUSTERING PROTEINS")
        print("=" * 60)

        # Check dependencies
        if not self.check_dependencies():
            return False

        # Count input proteins
        input_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
        print(f"\n📥 Input: {input_count} proteins từ {input_fasta}")

        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Prepare database
            db_dmnd = self.prepare_database(temp_path)

            # Run DIAMOND
            diamond_output = self.run_diamond_blastp(input_fasta, db_dmnd, temp_path)

            # Parse results
            diamond_df = self.parse_diamond_results(diamond_output)

            # Assign clusters
            results_df = self.assign_clusters(diamond_df, input_fasta)

        # Save results
        # Save full results with alignment info
        full_output = output_csv.replace('.csv', '_full.csv')
        results_df.to_csv(full_output, index=False)
        print(f"\n💾 Full results saved: {full_output}")

        # Save simplified version for update_database.py
        simplified_df = results_df[['protein_id', 'contig_id', 'keywords', 'cluster']]
        simplified_df.to_csv(output_csv, index=False)
        print(f"💾 Simplified results saved: {output_csv}")

        print("\n" + "=" * 60)
        print("✅ CLUSTERING HOÀN TẤT!")
        print("=" * 60)
        print("\n📌 Bước tiếp theo:")
        print(f"  python update_database.py --new_proteins {output_csv} --new_fasta {input_fasta}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Cluster proteins mới vào protein clusters hiện có của PhaTYP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python cluster_new_proteins.py --input new_proteins.fa --output new_proteins_clustered.csv

Script này sẽ:
  1. Align proteins mới với database hiện tại bằng DIAMOND
  2. Gán mỗi protein vào cluster của best hit
  3. Tạo file CSV để sử dụng với update_database.py

Yêu cầu:
  - DIAMOND phải được cài đặt (conda install -c bioconda diamond)
  - Input FASTA phải có format chuẩn với protein IDs và descriptions
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="File FASTA chứa proteins mới cần cluster"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file"
    )

    parser.add_argument(
        "--database_dir",
        default="database",
        help="Thư mục database (mặc định: database)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Số threads cho DIAMOND (mặc định: 8)"
    )

    args = parser.parse_args()

    # Check if input exists
    if not Path(args.input).exists():
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    # Run clustering
    clusterer = ProteinClusterer(args.database_dir, args.threads)
    success = clusterer.cluster(args.input, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
