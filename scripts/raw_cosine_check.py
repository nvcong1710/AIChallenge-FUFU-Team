"""Compare raw SigLIP cosine across queries — kiểm tra discriminability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.backend.services.retrieval import Retriever
from app.common.config import get_config
from app.common.encoder import SiglipEncoder


def main():
    cfg = get_config()
    enc = SiglipEncoder(cfg["models"]["siglip"], device=cfg["models"]["device"])
    ret = Retriever(cfg["storage"]["index_path"], cfg["storage"]["db_path"])

    queries = sys.argv[1:] or [
        "chơi cờ vua", "playing chess",
        "con mèo đang ngủ", "bãi biển hoàng hôn", "tắc đường",
    ]

    text_vecs = enc.encode_text(queries)
    print(f"\n{'Query':<32} {'Top1':<10} {'Top5 mean':<12} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    for q, qvec in zip(queries, text_vecs):
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
        scores = [s for _, s in ret.faiss_search(qvec, top_k=50)]
        if not scores:
            print(f"{q:<32} (empty index)")
            continue
        print(f"{q:<32} {scores[0]:<10.4f} {np.mean(scores[:5]):<12.4f} {min(scores):<10.4f} {max(scores):<10.4f}")


if __name__ == "__main__":
    main()
