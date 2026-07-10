import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from controller.base import Controller
from controller.eval import compute_eer_from_labels
from controller.trainer import Trainer


class DeepfakeLabelContractTest(unittest.TestCase):
    def test_perfect_bonafide_scores_have_zero_eer(self):
        labels = np.array([0, 0, 1, 1])
        bonafide_scores = np.array([0.01, 0.10, 0.90, 0.99])

        self.assertEqual(compute_eer_from_labels(labels, bonafide_scores), 0.0)

    def test_inverted_bonafide_scores_have_maximum_eer(self):
        labels = np.array([0, 0, 1, 1])
        inverted_scores = np.array([0.99, 0.90, 0.10, 0.01])

        self.assertEqual(compute_eer_from_labels(labels, inverted_scores), 1.0)

    def test_cm_score_is_bonafide_minus_spoof_logit(self):
        logits = torch.tensor([[4.0, -1.0], [-2.0, 3.0]])

        score = Controller._asvspoof_cm_score(None, logits)

        torch.testing.assert_close(score, torch.tensor([-5.0, 5.0]))

    def test_worse_eer_does_not_replace_best_checkpoint(self):
        trainer = Trainer.__new__(Trainer)
        trainer.dataset_name = "ASVspoof5"
        trainer.best_score = float("inf")
        trainer.logger = logging.getLogger("test")
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.wandb_run = None

        with tempfile.TemporaryDirectory() as work_dir:
            trainer.cfg = SimpleNamespace(
                general=SimpleNamespace(work_dir=work_dir),
                wandb=SimpleNamespace(enable=False),
            )
            with patch("torch.save") as save:
                trainer._save_checkpoint(metric=0.20)
                trainer._save_checkpoint(metric=0.40)

        self.assertEqual(trainer.best_score, 0.20)
        self.assertEqual(save.call_count, 1)


if __name__ == "__main__":
    unittest.main()
