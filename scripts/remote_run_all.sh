#!/usr/bin/env bash
# End-to-end v2 pipeline trên vast.ai:
#  - Full stack: SigLIP Large + Qwen-VL caption + EasyOCR + BGE-reranker
#  - Diverse data: COCO subset + Wikimedia + gTTS + BBB/Sintel
#  - Eval + HTML report
set -e

LOG=/root/bd/run.log
exec > >(tee "$LOG") 2>&1

cd /root/bd
source .venv/bin/activate

echo "===================== [STEP 1] Verify deps ====================="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import easyocr; print('easyocr OK')"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo ""

echo "===================== [STEP 2] Download models ====================="
python scripts/remote_download_models.py
echo ""

echo "===================== [STEP 3] Download test data ====================="
python scripts/download_test_data.py
python scripts/download_more_data.py
python scripts/download_vn_audio.py
python scripts/download_more_data_v2.py
python scripts/download_diverse_data.py
echo ""

echo "===================== [STEP 4] Ingest test-data (full stack) ====================="
du -sh test-data/
python -m app.ingest.cli test-data/
echo ""

echo "===================== [STEP 5] Search verify ====================="
for QUERY in "chơi cờ vua" "thị trường chứng khoán" "Lê Quang Liêm" "phở bò Hà Nội" "con thỏ trắng" "hoàng hôn Hà Nội" "công thức một"; do
    echo "--- '$QUERY' ---"
    python scripts/search_demo.py "$QUERY" 5 2>&1 | grep -E "^\[|score|caption|asr:|file:" | head -25
    echo ""
done

echo "===================== [STEP 6] Eval accuracy ====================="
python scripts/eval_accuracy.py 2>&1 | tail -40
echo ""

echo "===================== [STEP 7] Generate HTML report ====================="
python scripts/eval_html_report.py
echo ""

echo "===================== [STEP 8] Final stats ====================="
du -sh data/
ls -la data/
echo ""
echo "===================== ✓ DONE ====================="
