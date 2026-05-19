#!/usr/bin/env bash
# End-to-end pipeline: deps fix → models → data → ingest → search → stats.
set -e

LOG=/root/bd/run.log
exec > >(tee "$LOG") 2>&1

cd /root/bd
source .venv/bin/activate

echo "===================== [STEP 1] Patch deps ====================="
pip install --quiet protobuf tiktoken 'numpy<2'
python -c "import protobuf; print('protobuf', protobuf.__version__)" 2>/dev/null || echo "(protobuf installed)"
echo ""

echo "===================== [STEP 2] Download models ====================="
python scripts/remote_download_models.py
echo ""

echo "===================== [STEP 3] Download test data ====================="
python scripts/download_test_data.py
python scripts/download_more_data.py
python scripts/download_vn_audio.py
python scripts/download_more_data_v2.py
echo ""

echo "===================== [STEP 4] Ingest test-data ====================="
du -sh test-data/
python -m app.ingest.cli test-data/
echo ""

echo "===================== [STEP 5] Search verify ====================="
for QUERY in "chơi cờ vua" "thị trường chứng khoán" "Lê Quang Liêm" "phở bò Hà Nội" "con thỏ trắng"; do
    echo "--- '$QUERY' ---"
    python scripts/search_demo.py "$QUERY" 5 2>&1 | grep -E "^\[|score|caption|asr:|file:" | head -25
    echo ""
done

echo "===================== [STEP 6] Final stats ====================="
du -sh data/
ls -la data/
echo ""
echo "===================== ✓ DONE ====================="
