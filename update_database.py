#!/usr/bin/env python
"""
Script để cập nhật PhaTYP database với protein mới mà KHÔNG cần train lại model.

Cách sử dụng:
    python update_database.py --new_proteins new_proteins.csv --new_fasta new_proteins.fa

Input:
    - new_proteins.csv: File CSV với columns: protein_id,contig_id,keywords,cluster
    - new_proteins.fa: File FASTA chứa sequences của proteins mới

Lưu ý:
    - Protein cluster (PC_XXXXX) phải là một trong 45,578 clusters hiện có
    - Script sẽ validate clusters trước khi cập nhật
    - Backup tự động được tạo trước khi cập nhật
"""

import argparse
import pandas as pd
import gzip
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
import sys


class DatabaseUpdater:
    def __init__(self, database_dir="database"):
        self.database_dir = Path(database_dir)
        self.proteins_csv = self.database_dir / "proteins.csv"
        self.database_fa_gz = self.database_dir / "database.fa.gz"
        self.pc2wordsid_dict = self.database_dir / "pc2wordsid.dict"

        # Load existing data
        print("📂 Đang load database hiện tại...")
        self.existing_proteins = pd.read_csv(self.proteins_csv)

        with open(self.pc2wordsid_dict, 'rb') as f:
            self.valid_clusters = set(pickle.load(f).keys())

        print(f"✓ Database hiện có {len(self.existing_proteins)} proteins")
        print(f"✓ Có {len(self.valid_clusters)} protein clusters hợp lệ")

    def validate_new_proteins(self, new_proteins_df):
        """Kiểm tra dữ liệu protein mới"""
        print("\n🔍 Đang validate dữ liệu mới...")

        errors = []

        # Check required columns
        required_cols = ['protein_id', 'contig_id', 'keywords', 'cluster']
        missing_cols = set(required_cols) - set(new_proteins_df.columns)
        if missing_cols:
            errors.append(f"Thiếu columns: {missing_cols}")

        # Check for duplicate protein IDs with existing data
        existing_ids = set(self.existing_proteins['protein_id'])
        new_ids = set(new_proteins_df['protein_id'])
        duplicates = existing_ids & new_ids
        if duplicates:
            errors.append(f"Có {len(duplicates)} protein IDs trùng với database hiện tại")
            print(f"  ⚠️  Một số IDs trùng: {list(duplicates)[:5]}")

        # Check if clusters are valid (for non-empty clusters)
        non_empty_clusters = new_proteins_df[new_proteins_df['cluster'].notna()]
        invalid_clusters = set(non_empty_clusters['cluster']) - self.valid_clusters
        if invalid_clusters:
            errors.append(f"Có {len(invalid_clusters)} clusters không hợp lệ (không tồn tại trong 45,578 clusters)")
            print(f"  ⚠️  Một số clusters không hợp lệ: {list(invalid_clusters)[:5]}")
            print(f"  💡 Clusters hợp lệ phải có dạng PC_000000 đến PC_045577")

        if errors:
            print("\n❌ VALIDATION FAILED:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("✓ Validation thành công!")
        return True

    def validate_fasta(self, fasta_file, new_proteins_df):
        """Kiểm tra file FASTA"""
        print("\n🔍 Đang validate file FASTA...")

        fasta_ids = set()
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                fasta_ids.add(record.id)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file FASTA: {e}")
            return False

        csv_ids = set(new_proteins_df['protein_id'])

        missing_in_fasta = csv_ids - fasta_ids
        missing_in_csv = fasta_ids - csv_ids

        errors = []
        if missing_in_fasta:
            errors.append(f"Có {len(missing_in_fasta)} protein IDs trong CSV nhưng không có trong FASTA")
            print(f"  ⚠️  Một số IDs thiếu: {list(missing_in_fasta)[:5]}")

        if missing_in_csv:
            print(f"  ⚠️  Warning: Có {len(missing_in_csv)} sequences trong FASTA nhưng không có trong CSV (sẽ bỏ qua)")

        if errors:
            print("\n❌ FASTA VALIDATION FAILED:")
            for error in errors:
                print(f"  - {error}")
            return False

        print(f"✓ FASTA validation thành công! ({len(fasta_ids)} sequences)")
        return True

    def create_backup(self):
        """Tạo backup của database hiện tại"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.database_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)

        print(f"\n💾 Đang tạo backup tại {backup_dir}...")

        shutil.copy2(self.proteins_csv, backup_dir / "proteins.csv")
        shutil.copy2(self.database_fa_gz, backup_dir / "database.fa.gz")

        print("✓ Backup hoàn tất!")
        return backup_dir

    def update_proteins_csv(self, new_proteins_df):
        """Cập nhật proteins.csv"""
        print(f"\n📝 Đang cập nhật proteins.csv...")

        # Append new proteins
        updated_df = pd.concat([self.existing_proteins, new_proteins_df], ignore_index=True)

        # Save
        updated_df.to_csv(self.proteins_csv, index=False)

        print(f"✓ Đã thêm {len(new_proteins_df)} proteins mới")
        print(f"✓ Tổng số proteins: {len(updated_df)}")

    def update_database_fasta(self, new_fasta_file):
        """Cập nhật database.fa.gz"""
        print(f"\n📝 Đang cập nhật database.fa.gz...")

        # Create temporary uncompressed file
        temp_fa = self.database_dir / "database_temp.fa"

        # Decompress existing database
        with gzip.open(self.database_fa_gz, 'rt') as f_in:
            with open(temp_fa, 'w') as f_out:
                f_out.write(f_in.read())

        # Append new sequences
        new_count = 0
        with open(temp_fa, 'a') as f_out:
            for record in SeqIO.parse(new_fasta_file, "fasta"):
                SeqIO.write(record, f_out, "fasta")
                new_count += 1

        # Compress back
        with open(temp_fa, 'rb') as f_in:
            with gzip.open(self.database_fa_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Clean up
        temp_fa.unlink()

        print(f"✓ Đã thêm {new_count} sequences mới")

    def update(self, new_proteins_csv, new_fasta_file, skip_backup=False):
        """Main update function"""
        print("=" * 60)
        print("🚀 BẮT ĐẦU CẬP NHẬT DATABASE")
        print("=" * 60)

        # Load new data
        try:
            new_proteins_df = pd.read_csv(new_proteins_csv)
            print(f"✓ Đã load {len(new_proteins_df)} proteins mới từ CSV")
        except Exception as e:
            print(f"❌ Lỗi khi đọc file CSV: {e}")
            return False

        # Validate
        if not self.validate_new_proteins(new_proteins_df):
            return False

        if not self.validate_fasta(new_fasta_file, new_proteins_df):
            return False

        # Create backup
        if not skip_backup:
            backup_dir = self.create_backup()

        # Update
        try:
            self.update_proteins_csv(new_proteins_df)
            self.update_database_fasta(new_fasta_file)
        except Exception as e:
            print(f"\n❌ Lỗi khi cập nhật: {e}")
            if not skip_backup:
                print(f"💡 Khôi phục từ backup: {backup_dir}")
            return False

        print("\n" + "=" * 60)
        print("✅ CẬP NHẬT DATABASE THÀNH CÔNG!")
        print("=" * 60)
        print("\n📌 Lưu ý:")
        print("  - Model KHÔNG cần train lại")
        print("  - DIAMOND index sẽ tự động rebuild khi chạy preprocessing")
        print("  - Có thể test ngay bằng: python PhaTYP.py --contigs test.fa")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Cập nhật PhaTYP database với proteins mới (không cần train lại model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python update_database.py --new_proteins new_proteins.csv --new_fasta new_proteins.fa

Format file new_proteins.csv:
  protein_id,contig_id,keywords,cluster
  YP_123456.1,MyPhage_1,tail protein,PC_001234
  YP_123457.1,MyPhage_1,hypothetical protein,PC_005678

Lưu ý:
  - Cluster phải là một trong 45,578 PC clusters hiện có (PC_000000 - PC_045577)
  - Nếu protein không thuộc cluster nào, để trống cột cluster
  - File FASTA phải chứa sequences của tất cả protein IDs trong CSV
        """
    )

    parser.add_argument(
        "--new_proteins",
        required=True,
        help="File CSV chứa thông tin proteins mới"
    )

    parser.add_argument(
        "--new_fasta",
        required=True,
        help="File FASTA chứa sequences của proteins mới"
    )

    parser.add_argument(
        "--database_dir",
        default="database",
        help="Thư mục database (mặc định: database)"
    )

    parser.add_argument(
        "--skip_backup",
        action="store_true",
        help="Bỏ qua việc tạo backup (không khuyến nghị)"
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.new_proteins).exists():
        print(f"❌ File không tồn tại: {args.new_proteins}")
        sys.exit(1)

    if not Path(args.new_fasta).exists():
        print(f"❌ File không tồn tại: {args.new_fasta}")
        sys.exit(1)

    # Run update
    updater = DatabaseUpdater(args.database_dir)
    success = updater.update(args.new_proteins, args.new_fasta, args.skip_backup)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
