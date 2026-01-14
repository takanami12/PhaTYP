# Hướng dẫn cập nhật Database PhaTYP

## 📌 Tổng quan

Tài liệu này hướng dẫn cách cập nhật database của PhaTYP với proteins mới **mà KHÔNG cần train lại model**.

## 🎯 Khi nào sử dụng phương pháp này?

✅ **Sử dụng khi:**
- Bạn có thêm proteins mới từ các phages mới được công bố
- Muốn mở rộng database để tăng độ coverage
- Muốn cập nhật annotations/descriptions của proteins
- Muốn tránh việc train lại model (mất ~1 tuần với 4 GPUs)

❌ **KHÔNG sử dụng khi:**
- Bạn muốn thay đổi protein clustering algorithm
- Muốn tạo protein clusters hoàn toàn mới
- Muốn thay đổi vocabulary của model
- Trong trường hợp này, phải train lại model từ đầu

## 📋 Yêu cầu

### Phần mềm cần thiết:
```bash
# DIAMOND (cho protein alignment)
conda install -c bioconda diamond

# Python packages
pip install pandas biopython
```

### Dữ liệu cần chuẩn bị:
- File FASTA chứa protein sequences mới
- (Optional) Thông tin về contig/phage và protein descriptions

## 🚀 Quy trình cập nhật (3 bước)

### Bước 1: Cluster proteins mới vào clusters hiện có

Script `cluster_new_proteins.py` sẽ:
1. Align proteins mới với database hiện tại bằng DIAMOND
2. Gán mỗi protein vào cluster của best hit
3. Tạo file CSV phù hợp để cập nhật database

```bash
python cluster_new_proteins.py \
    --input new_proteins.fa \
    --output new_proteins_clustered.csv \
    --threads 8
```

**Input:** File FASTA với format:
```
>YP_123456.1 tail protein [Mycobacterium phage Example]
MTKPQTLNKIILLGAGFILGFQ...
>YP_123457.1 hypothetical protein [Mycobacterium phage Example]
MQTILNQILKGAGFILRFQ...
```

**Output:**
- `new_proteins_clustered.csv` - File đơn giản để dùng cho bước 2
- `new_proteins_clustered_full.csv` - File đầy đủ với thông tin alignment (identity, e-value, best hit)

**Output format:**
```csv
protein_id,contig_id,keywords,cluster
YP_123456.1,Mycobacterium_phage_Example,tail protein,PC_012345
YP_123457.1,Mycobacterium_phage_Example,hypothetical protein,PC_003456
```

**Lưu ý:**
- Proteins không có hit tốt sẽ có cluster rỗng (sẽ không được dùng trong prediction)
- Identity threshold mặc định: e-value < 1e-5
- Chỉ lấy best hit cho mỗi protein

### Bước 2: Kiểm tra kết quả clustering

Kiểm tra file `new_proteins_clustered_full.csv`:

```bash
# Xem proteins có cluster
python -c "
import pandas as pd
df = pd.read_csv('new_proteins_clustered_full.csv')
print(f'Tổng số proteins: {len(df)}')
print(f'Có cluster: {df[\"cluster\"].notna().sum()}')
print(f'Không có cluster: {df[\"cluster\"].isna().sum()}')
print(f'\nIdentity distribution:')
print(df['identity'].describe())
"
```

**Quyết định:**
- Proteins không có cluster sẽ không được dùng trong PhaTYP prediction
- Có thể filter theo identity threshold nếu muốn (ví dụ: chỉ giữ proteins với identity > 30%)

### Bước 3: Cập nhật database

Script `update_database.py` sẽ:
1. Validate dữ liệu (kiểm tra format, clusters hợp lệ, không trùng IDs)
2. Tạo backup của database hiện tại
3. Cập nhật `database/proteins.csv` và `database/database.fa.gz`

```bash
python update_database.py \
    --new_proteins new_proteins_clustered.csv \
    --new_fasta new_proteins.fa
```

**Output:**
```
🚀 BẮT ĐẦU CẬP NHẬT DATABASE
============================================================
📂 Đang load database hiện tại...
✓ Database hiện có 431582 proteins
✓ Có 45578 protein clusters hợp lệ

🔍 Đang validate dữ liệu mới...
✓ Validation thành công!

🔍 Đang validate file FASTA...
✓ FASTA validation thành công! (1500 sequences)

💾 Đang tạo backup tại database/backup_20260114_143022...
✓ Backup hoàn tất!

📝 Đang cập nhật proteins.csv...
✓ Đã thêm 1500 proteins mới
✓ Tổng số proteins: 433082

📝 Đang cập nhật database.fa.gz...
✓ Đã thêm 1500 sequences mới

============================================================
✅ CẬP NHẬT DATABASE THÀNH CÔNG!
============================================================

📌 Lưu ý:
  - Model KHÔNG cần train lại
  - DIAMOND index sẽ tự động rebuild khi chạy preprocessing
  - Có thể test ngay bằng: python PhaTYP.py --contigs test.fa
```

## ✅ Kiểm tra sau khi cập nhật

### Test với phage mới:

```bash
# Test prediction
python PhaTYP.py --contigs your_new_phage.fa --out predictions.csv

# Kiểm tra kết quả
cat predictions.csv
```

### Kiểm tra database size:

```bash
# Check số proteins
wc -l database/proteins.csv

# Check database size
ls -lh database/database.fa.gz
```

## 🔧 Troubleshooting

### Lỗi: "DIAMOND không được tìm thấy"
```bash
# Cài đặt DIAMOND
conda install -c bioconda diamond

# Hoặc
wget http://github.com/bbuchfink/diamond/releases/download/v0.9.14/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
# Copy diamond binary vào $PATH
```

### Lỗi: "Có clusters không hợp lệ"
```bash
# Kiểm tra clusters trong file
python -c "
import pandas as pd
df = pd.read_csv('new_proteins_clustered.csv')
invalid = df[~df['cluster'].str.match(r'PC_\d{6}', na=False)]
print('Invalid clusters:', invalid['cluster'].unique())
"

# Clusters hợp lệ: PC_000000 đến PC_045577
```

### Lỗi: "Protein IDs trùng với database hiện tại"
```bash
# Tìm IDs trùng
python -c "
import pandas as pd
existing = pd.read_csv('database/proteins.csv')
new = pd.read_csv('new_proteins_clustered.csv')
duplicates = set(existing['protein_id']) & set(new['protein_id'])
print(f'Trùng {len(duplicates)} IDs:', list(duplicates)[:10])
"

# Giải pháp: Rename protein IDs hoặc loại bỏ duplicates
```

### Khôi phục từ backup (nếu có lỗi):
```bash
# List backups
ls -lt database/backup_*/

# Restore từ backup gần nhất
BACKUP_DIR=$(ls -td database/backup_*/ | head -1)
cp $BACKUP_DIR/proteins.csv database/
cp $BACKUP_DIR/database.fa.gz database/

echo "✓ Đã khôi phục từ $BACKUP_DIR"
```

## 📊 Thống kê Database hiện tại

```bash
# Quick stats
python -c "
import pandas as pd
import pickle

# Proteins
df = pd.read_csv('database/proteins.csv')
print(f'Total proteins: {len(df):,}')
print(f'With clusters: {df[\"cluster\"].notna().sum():,}')
print(f'Without clusters: {df[\"cluster\"].isna().sum():,}')

# Clusters
with open('database/pc2wordsid.dict', 'rb') as f:
    pc2id = pickle.load(f)
print(f'Total clusters: {len(pc2id):,}')
print(f'Cluster range: {min(pc2id.keys())} to {max(pc2id.keys())}')
"
```

## 🎓 Hiểu về Protein Clusters

### Protein Cluster là gì?

Protein clusters (PC) là các nhóm proteins tương đồng về sequence và chức năng. PhaTYP sử dụng:
- **45,578 clusters** (PC_000000 đến PC_045577)
- Mỗi cluster đại diện cho một "từ" trong vocabulary của BERT model
- Tương tự như word2vec trong NLP, protein2vec trong PhaTYP

### Tại sao phải map vào clusters hiện có?

BERT model đã được train với vocabulary cố định 45,578 tokens. Mỗi token có:
- **Embedding vector**: Vector đại diện ý nghĩa của protein cluster
- **Attention weights**: Trọng số học được từ 200K training steps

Nếu thêm cluster mới (PC_045578, ...) thì:
- ❌ Model không có embedding cho token mới
- ❌ Cần train lại toàn bộ (~1 tuần, 4 GPUs)
- ❌ Mất tất cả knowledge đã học

Nếu map vào clusters hiện có:
- ✅ Sử dụng embeddings đã train sẵn
- ✅ Không cần train lại
- ✅ Chỉ mất vài phút để cập nhật

### Làm sao biết mapping có tốt không?

Kiểm tra identity của alignment:
```bash
# Check identity distribution
python -c "
import pandas as pd
df = pd.read_csv('new_proteins_clustered_full.csv')
print('Identity statistics:')
print(df['identity'].describe())
print('\nIdentity ranges:')
print(f'>90%: {(df[\"identity\"] > 90).sum()}')
print(f'70-90%: {((df[\"identity\"] >= 70) & (df[\"identity\"] <= 90)).sum()}')
print(f'50-70%: {((df[\"identity\"] >= 50) & (df[\"identity\"] < 70)).sum()}')
print(f'<50%: {(df[\"identity\"] < 50).sum()}')
"
```

**Guidelines:**
- Identity > 70%: Excellent match
- Identity 50-70%: Good match (distant homologs)
- Identity 30-50%: Acceptable (same protein family)
- Identity < 30%: Questionable (might not be reliable)

## 📚 Tham khảo thêm

- **PhaTYP Paper**: [Link to paper]
- **DIAMOND Manual**: https://github.com/bbuchfink/diamond
- **Original database**: Từ RefSeq database (142K phages cho pre-training)

## ❓ FAQ

**Q: Có thể thêm bao nhiêu proteins mới?**
A: Không giới hạn. Chỉ cần có đủ disk space cho database.

**Q: Proteins không có hit tốt thì sao?**
A: Sẽ có cluster rỗng, không được sử dụng trong prediction. Điều này là bình thường với novel proteins.

**Q: Có cần chạy lại PhaTYP prediction cho data cũ không?**
A: Không. Results cũ vẫn hợp lệ. Chỉ phages mới có thể được dự đoán tốt hơn nhờ database mở rộng.

**Q: Có thể update descriptions/keywords của proteins hiện có không?**
A: Có. Sửa trực tiếp trong `database/proteins.csv`, nhưng backup trước.

**Q: DIAMOND alignment chạy chậm quá?**
A: Tăng số threads: `--threads 32`. Hoặc sử dụng sensitive mode: thêm `--sensitive` (chậm hơn nhưng chính xác hơn).

**Q: Làm sao để cập nhật định kỳ với proteins mới từ RefSeq?**
A: Có thể tạo cron job để:
1. Download proteins mới từ RefSeq
2. Chạy clustering script
3. Cập nhật database tự động

## 💡 Best Practices

1. **Luôn backup** trước khi cập nhật (script tự động làm điều này)
2. **Validate alignment results** trước khi cập nhật database
3. **Test với sample data** sau khi cập nhật
4. **Document changes**: Ghi lại những gì đã thêm vào database
5. **Version control**: Có thể git commit database changes (nếu size cho phép)

## 🎉 Kết luận

Quy trình 3 bước này cho phép bạn:
- ✅ Cập nhật database nhanh chóng (vài phút thay vì 1 tuần)
- ✅ Không cần GPU
- ✅ Không mất knowledge đã học
- ✅ Mở rộng coverage cho phages mới

Chúc may mắn với PhaTYP! 🦠🔬
