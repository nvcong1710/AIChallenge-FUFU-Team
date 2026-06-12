# Vast.ai run guide — Full stack v2

## Recommended hardware

| GPU | VRAM | Price/h | Note |
|---|---|---|---|
| **RTX 4090** | 24GB | **$0.40-0.60** | ✅ Best balance — fits all models tight, fastest ingest |
| RTX 3090 | 24GB | $0.30-0.45 | OK, ~30% chậm hơn 4090 |
| A6000 | 48GB | $0.50-0.80 | Comfortable, dư memory cho experiment |
| A5000 | 24GB | $0.30 | Workstation, OK |
| A100 40GB | 40GB | $1-1.50 | Overkill, không cần |

VRAM cần lúc ingest **đồng thời** load (peak):
- SigLIP-2 Large (fp16): 1.2 GB
- NLLB-200 (fp16): 1.3 GB
- Qwen2.5-3B paraphrase (INT4): 2.5 GB
- Qwen2.5-VL-7B caption (INT4): 5 GB
- PhoWhisper-medium (fp16): 3 GB
- YOLO-World v8l: 1.5 GB
- BGE-reranker-v2-m3 (fp16): 2.5 GB
- EasyOCR: 1 GB
- Activation buffer: 3-5 GB
- **Tổng: ~21-22 GB** → 4090/3090 24GB tight, A6000 thoải mái

## Filter Vast.ai search

```
GPU:        RTX 4090 (24GB) hoặc A6000 (48GB)
CUDA:       ≥ 12.1
Disk:       ≥ 100 GB                  ← models ~25GB, data ~2GB, working ~10GB
Inet down:  ≥ 200 Mbps                ← HF download 25GB
Reliability: > 99%
```

Sort theo **$/DLPerf** (best perf-per-dollar).

## Cost estimate cho 1 session (4090)

| Phase | Time | Cost |
|---|---|---|
| Setup deps | 5 phút | $0.04 |
| Download models (~25GB) | 15-25 phút | $0.20 |
| Download test data | 10-15 phút | $0.10 |
| Ingest (full stack, Qwen-VL bottleneck) | 60-90 phút | $0.60-0.80 |
| Eval + report | 5-10 phút | $0.07 |
| **Tổng** | **~2 giờ** | **~$1.10** |

## Workflow

### 1. Rent instance trên vast.ai

- Template: **PyTorch 2.4 CUDA 12.1** (hoặc tương đương)
- Disk: 100 GB
- Mở port: 8080 (FastAPI) hoặc dùng SSH tunnel
- Khi instance ready, vast.ai cho lệnh SSH dạng:
  ```
  ssh -p <port> root@<host>
  ```

### 2. SSH + clone từ GitHub

```bash
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host>

# Trên server:
apt update && apt install -y tmux htop git ffmpeg libgl1 libglib2.0-0
tmux new -s work

git clone https://github.com/nvcong1710/AIChallenge-FUFU-Team.git /root/bd
cd /root/bd

# Copy production settings (nếu khác)
cp scripts/settings_remote.yaml config/settings.yaml
```

### 3. Setup env + download models + ingest

```bash
chmod +x scripts/remote_setup.sh scripts/remote_run_all.sh
./scripts/remote_setup.sh                # ~5 phút

# (Optional) Setup Kaggle API để tải MSR-VTT benchmark
# 1) Lấy kaggle.json từ https://www.kaggle.com/settings → API → Create New Token
# 2) Upload lên server: scp -P <port> kaggle.json root@<host>:/root/.kaggle/
mkdir -p /root/.kaggle
# Paste kaggle.json content vào /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json

# Run pipeline đầy đủ trong background:
nohup ./scripts/remote_run_all.sh > /dev/null 2>&1 < /dev/null &
disown

# Theo dõi log:
tail -f /root/bd/run.log
```

**MSR-VTT benchmark** (gold standard cho text→video retrieval):
- 10K video clips × 20 captions = 200K query annotations
- Đã thiết lập subset 200 videos × 3 queries = 600 test cases EN + 600 VN dịch
- Recall@K so sánh trực tiếp được với literature (CLIP, BLIP-2 papers...)
- Nếu skip Kaggle setup, hệ vẫn chạy custom VN test cases (37 query)

### 4. Pull results về local

```bash
# Trên server:
tar -czf /tmp/bd_data.tar.gz data/ eval_report.html run.log

# Local máy anh:
scp -P <port> root@<host>:/tmp/bd_data.tar.gz ./remote_results/
tar -xzf remote_results/bd_data.tar.gz -C ./

# Xem HTML report
py -3.10 -m http.server 8765
# Mở http://localhost:8765/eval_report.html
```

### 5. Destroy instance khi xong

Vast.ai dashboard → instance → **Destroy** (không "Stop" — Stop vẫn tính tiền disk).

## Troubleshooting

| Issue | Fix |
|---|---|
| `transformers 5.x` install thay 4.50 | Pin chính xác: `pip install 'transformers==4.50.0'` |
| SigLIP-2 tokenizer error | Đã patch `use_fast=True` trong encoder.py + remote_download_models.py |
| protobuf missing | Đã add vào remote_setup.sh |
| Qwen-VL OOM | Đảm bảo `caption_quant_4bit: true` trong settings.yaml |
| Disk full | Giảm test data: bỏ download Sintel (642MB) khỏi download scripts |
| ASR Sintel chậm 5+ phút | Bình thường vì Sintel có nhiều dialogue. Có thể bỏ qua nếu vội. |

## Tips

- **Tmux**: luôn dùng `tmux new -s work` để không mất session khi ssh disconnect
- **Background ingest**: `nohup ... &` + `disown` → script chạy qua sess
- **Test local trước**: chạy `python scripts/eval_html_report.py` trên data đã có để verify trước khi rent
- **Stop instance ngay khi xong**: Vast tính tiền theo giây
