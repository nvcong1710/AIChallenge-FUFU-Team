"""Unit tests cho app.eval.metrics — chấm điểm ĐÚNG theo HCM AI Challenge 2025.

Metric chính thức (đã verify, xem PHU-LUC-KY-THUAT-2026.md §1.1):
    "Mean of Top-k R-Scores", trung bình qua k ∈ {1, 5, 20, 50, 100}.

Chạy: python -m unittest tests.test_eval_metrics   (KHÔNG cần pytest/numpy).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.metrics import (  # noqa: E402
    K_VALUES,
    aggregate,
    mean_top_k_r_score,
    r_score_kis,
    r_score_trake,
    r_score_vqa,
)


class TestMeanTopKRScore(unittest.TestCase):
    def test_k_values_are_competition_spec(self):
        self.assertEqual(K_VALUES, (1, 5, 20, 50, 100))

    def test_correct_at_rank_1_scores_full(self):
        # item đúng ở rank 1 (R-score 1.0) → trúng mọi ngưỡng k → điểm = 1.0
        rscores = [1.0] + [0.0] * 99
        self.assertAlmostEqual(mean_top_k_r_score(rscores), 1.0)

    def test_correct_at_rank_6_misses_k1_and_k5(self):
        # item đúng ở rank 6 (index 5): hit@1=0, hit@5=0, hit@20=1, hit@50=1, hit@100=1
        # → (0+0+1+1+1)/5 = 0.6
        rscores = [0.0] * 5 + [1.0] + [0.0] * 94
        self.assertAlmostEqual(mean_top_k_r_score(rscores), 0.6)

    def test_correct_at_rank_21_only_k50_k100(self):
        # rank 21 (index 20): chỉ k=50,100 trúng → 2/5 = 0.4
        rscores = [0.0] * 20 + [1.0]
        self.assertAlmostEqual(mean_top_k_r_score(rscores), 0.4)

    def test_empty_results_score_zero(self):
        self.assertEqual(mean_top_k_r_score([]), 0.0)

    def test_uses_prefix_max_for_continuous_scores(self):
        # TRAKE: điểm liên tục. rank1=0.5, rank3=1.0.
        # max(top1)=0.5; max(top5)=1.0; max(top20)=1.0; max(top50)=1.0; max(top100)=1.0
        # → (0.5 + 1 + 1 + 1 + 1)/5 = 0.9
        rscores = [0.5, 0.0, 1.0]
        self.assertAlmostEqual(mean_top_k_r_score(rscores), 0.9)

    def test_fewer_than_k_results_ok(self):
        # chỉ 3 kết quả, đúng ở rank 2 → mọi k≥2 trúng; k=1 trượt → (0+1+1+1+1)/5=0.8
        rscores = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(mean_top_k_r_score(rscores), 0.8)


class TestRScoreKIS(unittest.TestCase):
    def test_video_match_and_frame_in_range_scores_1(self):
        self.assertEqual(r_score_kis("L01_V001.mp4", 13.0, "L01_V001", (12.5, 18.0)), 1.0)

    def test_frame_out_of_range_scores_0(self):
        self.assertEqual(r_score_kis("L01_V001.mp4", 30.0, "L01_V001", (12.5, 18.0)), 0.0)

    def test_wrong_video_scores_0(self):
        self.assertEqual(r_score_kis("L02_V099.mp4", 13.0, "L01_V001", (12.5, 18.0)), 0.0)

    def test_frame_on_boundary_inclusive(self):
        self.assertEqual(r_score_kis("L01_V001", 12.5, "L01_V001", (12.5, 18.0)), 1.0)
        self.assertEqual(r_score_kis("L01_V001", 18.0, "L01_V001", (12.5, 18.0)), 1.0)


class TestRScoreVQA(unittest.TestCase):
    def test_all_correct_scores_1(self):
        self.assertEqual(
            r_score_vqa("L01_V001", 13.0, "Hà Nội", "L01_V001", (12.5, 18.0), "Hà Nội"), 1.0
        )

    def test_answer_mismatch_scores_0(self):
        self.assertEqual(
            r_score_vqa("L01_V001", 13.0, "Sài Gòn", "L01_V001", (12.5, 18.0), "Hà Nội"), 0.0
        )

    def test_answer_match_is_case_and_whitespace_insensitive(self):
        self.assertEqual(
            r_score_vqa("L01_V001", 13.0, "  hà nội ", "L01_V001", (12.5, 18.0), "Hà Nội"), 1.0
        )

    def test_correct_answer_wrong_frame_scores_0(self):
        self.assertEqual(
            r_score_vqa("L01_V001", 99.0, "Hà Nội", "L01_V001", (12.5, 18.0), "Hà Nội"), 0.0
        )


class TestRScoreTRAKE(unittest.TestCase):
    def test_all_events_hit_scores_1(self):
        events = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]
        frames = [2.0, 12.0, 23.0]
        self.assertAlmostEqual(r_score_trake("L01_V001", frames, "L01_V001", events), 1.0)

    def test_half_events_hit(self):
        events = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0), (30.0, 35.0)]
        frames = [2.0, 99.0, 23.0, 99.0]  # event 1 & 3 trúng → 2/4 = 0.5
        self.assertAlmostEqual(r_score_trake("L01_V001", frames, "L01_V001", events), 0.5)

    def test_wrong_video_scores_0(self):
        events = [(0.0, 5.0)]
        self.assertEqual(r_score_trake("OTHER", [2.0], "L01_V001", events), 0.0)

    def test_missing_frames_count_as_miss(self):
        events = [(0.0, 5.0), (10.0, 15.0)]
        frames = [2.0]  # thiếu frame cho event 2 → 1/2 = 0.5
        self.assertAlmostEqual(r_score_trake("L01_V001", frames, "L01_V001", events), 0.5)


class TestAggregate(unittest.TestCase):
    def test_overall_and_per_task_means(self):
        per_query = [
            {"task": "kis", "score": 1.0},
            {"task": "kis", "score": 0.0},
            {"task": "trake", "score": 0.5},
        ]
        agg = aggregate(per_query)
        self.assertAlmostEqual(agg["overall"], (1.0 + 0.0 + 0.5) / 3)
        self.assertAlmostEqual(agg["by_task"]["kis"], 0.5)
        self.assertAlmostEqual(agg["by_task"]["trake"], 0.5)
        self.assertEqual(agg["n"], 3)

    def test_empty_aggregate_is_zero(self):
        agg = aggregate([])
        self.assertEqual(agg["overall"], 0.0)
        self.assertEqual(agg["n"], 0)


if __name__ == "__main__":
    unittest.main()
