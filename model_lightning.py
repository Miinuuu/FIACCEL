from typing import Any, List, Optional, Tuple, Union
import hydra
import pytorch_lightning
import torch
import torch.nn.functional as F
import torch.nn as nn
from datamodules.video_data_api import VideoData
from model_pipeline import FIACCELPipeline
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio
from utils.hydra_tools import OmegaConf  # ! Needed for resolvets to take effect
from model.loss import LapLoss


class FIACCELModule(pytorch_lightning.LightningModule):
    """Encapsulates model + loss + optimizer + metrics in a PL module"""

    def __init__(self, model: FIACCELPipeline, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg_optim = getattr(cfg, "optim", None)
        self.cfg_scheduler = getattr(cfg, "scheduler", None)
        self.lr_annealing_frequency = getattr(
            getattr(cfg, "training_loop", None), "lr_annealing_frequency", None
        )
        self.laploss = LapLoss()
        self.L1_loss = nn.L1Loss()
        self.metrics_every_n_steps = int(
            getattr(cfg.training_loop, "metrics_every_n_steps", 1)
        )
        if self.metrics_every_n_steps < 1:
            raise ValueError("metrics_every_n_steps must be positive")
    def forward(self, batch: VideoData) -> Tensor:
        return self.model(batch)

    def training_step(
        self, batch, batch_idx: int
    ) -> Tensor:
        pred = self(batch)
        gt = batch.video_tensor[:, 1]
        l1_loss = self.L1_loss(pred, gt)
        # Only L1 contributes gradients. Compute diagnostics at logging steps,
        # without building an unused Laplacian loss graph or syncing every step.
        if (self.global_step + 1) % self.metrics_every_n_steps == 0:
            with torch.no_grad():
                self.log_dict(
                    {
                        "mse_loss": F.mse_loss(pred, gt),
                        "l1_loss": l1_loss.detach(),
                        "laploss": self.laploss(pred, gt),
                        "PSNR": peak_signal_noise_ratio(
                            pred, gt, data_range=1.0, dim=(1, 2, 3)
                        ),
                        "PSNR_int": peak_signal_noise_ratio(
                            (pred * 255 + 0.5).to(torch.uint8),
                            (gt * 255 + 0.5).to(torch.uint8),
                            data_range=255, dim=(1, 2, 3),
                        ),
                    },
                    sync_dist=True, on_step=True, on_epoch=False,
                    batch_size=gt.shape[0],
                )
        return l1_loss

    def validation_step(self, batch, batch_idx) -> None:
        pred = self(batch)
        gt = batch.video_tensor[:, 1]

        self.log_dict(
            {
                "mse_loss": F.mse_loss(pred, gt),
                "val_PSNR": peak_signal_noise_ratio(
                    pred, gt, data_range=1.0, dim=(1, 2, 3)
                ),
                "val_PSNR_int": peak_signal_noise_ratio(
                    (pred * 255 + 0.5).to(torch.uint8),
                    (gt * 255 + 0.5).to(torch.uint8),
                    data_range=255,
                    dim=(1, 2, 3),
                ),
            },
            sync_dist=True, on_step=False, on_epoch=True,
            batch_size=gt.shape[0],
        )

    def test_step(
        self,
        batch: VideoData,
        batch_idx: int,
        run_fwd: bool = True,
    ):
        B, T, C, H, W = batch.shape
        assert B == 1, "Metrics calculation only supported for batch size 1."
        assert self.training is False

        with torch.no_grad():
            
            imgs=batch.video_tensor
            B, T, C, H, W = imgs.shape
            img0,gt,img1 = imgs[:, 0,...],imgs[:,1,...],imgs[:, 2,...]
            video_ref_uint8 = (gt * 255 + 0.5).to(torch.uint8)
            pred = self.model.inference(batch)
            
            # PSNR
            # caveat: real compression people use YCbCr
            recon_uint8 = (pred * 255 + 0.5).to(torch.uint8)
            psnr = peak_signal_noise_ratio(
                recon_uint8, video_ref_uint8, data_range=255
            ).item()
  
            print(psnr)
            self.log_dict(
                {f"psnr_int": psnr, 
                 },
                sync_dist=True,
            )
        return psnr

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,  # Single optimizer
        Tuple[torch.optim.Optimizer, torch.optim.Optimizer],  # Tuple or list of optim
        List[torch.optim.Optimizer],
        dict,  # "optimizer" key, and (optionally) an "lr_scheduler"
        Any,  # 2 lists: first with optimizers, second has LR schedulers; or Tuple[Dict]
    ]:
        # PL allows return type to be tuple/list/dict/two lists/tuple of dicts/None
        model_params = (
            p
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        )

        base_optim = hydra.utils.instantiate(self.cfg_optim, params=model_params)
        base_scheduler = hydra.utils.instantiate(
            self.cfg_scheduler, optimizer=base_optim
        )
        scheduler = {
            "scheduler": base_scheduler,
            "interval": "step",
            "frequency": self.lr_annealing_frequency,
        }
        return {"optimizer": base_optim, "lr_scheduler": scheduler}
