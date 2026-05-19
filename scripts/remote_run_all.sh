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
# MSR-VTT (Kaggle gold-standard benchmark): cần ~/.kaggle/kaggle.json
if [ -f ~/.kaggle/kaggle.json ]; then
    pip install --quiet kaggle
    python scripts/download_msrvtt.py || echo "  ⚠ MSR-VTT download fail, skip"
else
    echo "  ⚠ Kaggle API key chưa setup → skip MSR-VTT. Đặt kaggle.json vào ~/.kaggle/ rồi rerun."
fi
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
echo "--- Custom VN test cases ---"
python scripts/eval_accuracy.py 2>&1 | tail -40
echo ""
if [ -f scripts/test_cases_msrvtt.json ]; then
    echo "--- MSR-VTT benchmark (gold standard, EN queries) ---"
    python scripts/eval_accuracy.py --cases scripts/test_cases_msrvtt.json 2>&1 | tail -20
    # Sinh VN version + eval
    python scripts/translate_msrvtt_to_vn.py 2>&1 | tail -5 || true
    if [ -f scripts/test_cases_msrvtt_vn.json ]; then
        echo "--- MSR-VTT translated VN (cross-lingual test) ---"
        python scripts/eval_accuracy.py --cases scripts/test_cases_msrvtt_vn.json 2>&1 | tail -20
    fi
fi
echo ""

echo "===================== [STEP 7] Generate HTML reports ====================="
python scripts/eval_html_report.py
if [ -f scripts/test_cases_msrvtt.json ]; then
    mv eval_report.html eval_report_vn.html 2>/dev/null || true
    python scripts/eval_html_report.py --cases scripts/test_cases_msrvtt.json
    mv eval_report.html eval_report_msrvtt.html 2>/dev/null || true
fi
echo ""

echo "===================== [STEP 8] Final stats ====================="
du -sh data/
ls -la data/
echo ""
echo "===================== ✓ DONE ====================="
