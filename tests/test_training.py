"""Verify sampled diagnostics preserve QAT gradients and validation weighting."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from datamodules.video_data_api import VideoDataset
from model.flow_estimation import flow_estimation_int
from model_lightning import FIACCELModule
from model_pipeline import FIACCELPipeline


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.batch = VideoDataset(torch.rand(3, 3, 3, 64, 64))
        cfg = OmegaConf.create({"training_loop": {"metrics_every_n_steps": 100}})
        self.module = FIACCELModule(FIACCELPipeline(flow_estimation_int()), cfg)
        self.module._trainer = SimpleNamespace(global_step=0)

    def test_sampled_diagnostics_preserve_qat_loss_and_gradients(self):
        reference = copy.deepcopy(self.module.model)
        expected = F.l1_loss(reference(self.batch), self.batch.video_tensor[:, 1])
        expected.backward()
        with patch.object(self.module, "log_dict") as log, patch.object(
            self.module.laploss, "forward", side_effect=AssertionError("unused diagnostic")
        ):
            actual = self.module.training_step(self.batch, 0)
            actual.backward()
            log.assert_not_called()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in self.module.model.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)

    def test_diagnostics_only_at_requested_step_and_without_gradients(self):
        self.module._trainer.global_step = 99
        with patch.object(self.module, "log_dict") as log:
            loss = self.module.training_step(self.batch, 99)
        self.assertTrue(loss.requires_grad)
        metrics = log.call_args.args[0]
        self.assertEqual(set(metrics), {"mse_loss", "l1_loss", "laploss", "PSNR", "PSNR_int"})
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        self.assertTrue(log.call_args.kwargs["sync_dist"])

    def test_validation_batching_preserves_mean_per_image_psnr(self):
        with torch.no_grad():
            self.module(self.batch)  # Calibrate QAT activation scales.
        self.module.eval()
        with torch.no_grad(), patch.object(self.module, "log_dict") as log:
            self.module.validation_step(self.batch, 0)
            batched = log.call_args.args[0]
            self.assertEqual(log.call_args.kwargs["batch_size"], 3)
            self.assertFalse(log.call_args.kwargs["on_step"])
            self.assertTrue(log.call_args.kwargs["on_epoch"])
            single = []
            for index in range(3):
                self.module.validation_step(VideoDataset(self.batch.video_tensor[index:index+1]), index)
                single.append(log.call_args.args[0])
        for key, value in batched.items():
            torch.testing.assert_close(value, torch.stack([row[key] for row in single]).mean(),
                                       rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
