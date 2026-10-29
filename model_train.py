import hashlib
from pathlib import Path
from typing import Dict
import hydra
import pytorch_lightning as pl
import torch
import torch.distributed
import wandb
from datamodules.video_data_api import VideoData, VideoDataset
from model_lightning import FIACCELModule
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torchvision.utils import make_grid
from utils.hydra_tools import OmegaConf
import numpy as np
import os
import time
#os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class TrainingStatusCallback(pl.Callback):
    """Keep detached training logs readable without a terminal progress bar."""

    def on_train_start(self, trainer, pl_module):
        self.started_at = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if trainer.is_global_zero and step % trainer.log_every_n_steps == 0:
            loss = trainer.callback_metrics.get("l1_loss", float("nan"))
            psnr = trainer.callback_metrics.get("PSNR", float("nan"))
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"step={step} l1_loss={float(loss):.6f} PSNR={float(psnr):.3f} "
                  f"lr={lr:.6g} elapsed_s={time.monotonic()-self.started_at:.1f}", flush=True)

    def on_validation_start(self, trainer, pl_module):
        self.validation_started_at = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            psnr = trainer.callback_metrics.get("val_PSNR", float("nan"))
            print(f"validation step={trainer.global_step} val_PSNR={float(psnr):.3f} "
                  f"validation_s={time.monotonic()-self.validation_started_at:.1f}", flush=True)


class WandbImageCallback(pl.Callback):
    """
    Log images at end of each validatin step
    """

    def __init__(self, train_batch: VideoData, eval_batch: VideoData):
        super().__init__()
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.key_set = ("image0","image1","image2","image_interpolated")

    def log_images(
        self, trainer, base_key: str, image_dict: Dict[str, Tensor], global_step: int
    ) -> None:
        for key in self.key_set:
            if image_dict.get(key) is not None:
                log_dict = {
                    f"{base_key}/{key}": wandb.Image(image_dict[key], caption=f"{key}"),
                    "global_step": global_step,
                }
                trainer.logger.experiment.log(log_dict)

    def _compute_input_output_batch(
        self, pl_module: pl.LightningModule, batch: VideoData
    ):
        batch = VideoDataset(
            video_tensor=(batch.video_tensor).to(device=pl_module.device)
        )
        pred = pl_module(batch)
        idx = np.random.randint(0, batch.video_tensor.shape[0])
        return {
            "image0": make_grid(batch.video_tensor[idx,0,...], nrow=7),
            "image_interpolated": make_grid(pred, nrow=7),
            "image1": make_grid(batch.video_tensor[idx,1,...], nrow=7),
            "image2": make_grid(batch.video_tensor[idx,2,...], nrow=7),
        }

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self.log_images(
            trainer,
            "train_images",
            self._compute_input_output_batch(pl_module, self.train_batch),
            pl_module.global_step,
        )
        self.log_images(
            trainer,
            "eval_images",
            self._compute_input_output_batch(pl_module, self.eval_batch),
            pl_module.global_step,
        )


def build_image_logger(data_module: LightningDataModule):
    data_module.setup(stage=None)  # type: ignore

    train_sample = next(iter(data_module.train_dataloader()))
    val_sample = next(iter(data_module.val_dataloader()))
    return WandbImageCallback(train_sample, val_sample)  # type: ignore


@hydra.main(version_base="1.3", config_path="config", config_name="train_config_float")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    ######################################################################################
    # Check for saved checkpoints
    save_dir = Path(cfg.checkpoint.callback.dirpath).expanduser().absolute()
    save_dir.mkdir(exist_ok=True, parents=True)

    if (
        not cfg.checkpoint.overwrite
        and not cfg.checkpoint.resume_training
        and len(list(save_dir.glob("*.ckpt"))) > 0
    ):
        raise RuntimeError(
            "Checkpoints detected in save directory: set resume_training=True"
            " to restore trainer state from these checkpoints, or set overwrite=True"
            " to ignore them."
        )

    last_checkpoint = save_dir / "last.ckpt"
   
    if last_checkpoint.exists() and cfg.checkpoint.resume_training:
        print(f"Resuming training from last checkpoint = {last_checkpoint}.")
    elif cfg.checkpoint.resume_training:
        raise FileNotFoundError(f"Resume checkpoint not found: {last_checkpoint}")
    else:
        print(f"Initialising new model.")
        last_checkpoint = None

    # set up logger
    log_dir = Path.cwd().absolute() / "wandb_logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    # This will create an id based on the logging path
    name = "/".join([Path.cwd().parent.name, Path.cwd().name])
    sha = hashlib.sha256()
    sha.update(str(Path.cwd()).encode())
    wandb_id = sha.hexdigest()

    wandb_logger = WandbLogger(
        name=name,
        save_dir=str(log_dir),
        id=wandb_id,
        config=OmegaConf.to_container(cfg, resolve=True),  #! resolve=True to load later
        **cfg.logger,
    )
    ######################################################################################

    ### Instantiate dataloader module, model module and set up a wandb watch ###
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, pin_memory=cfg.ngpu != 0
    )
    # isntantiate model outside the PLModule for ease of debugging
    #print(cfg.model)
    model = hydra.utils.instantiate(cfg.model)
    modelmodule = FIACCELModule(model, cfg=cfg)
    #print(modelmodule)
    # wandb_logger.watch(model=modelmodule.model.bottleneck, log="all", log_freq=100)

    ### Set up trainer and fit the model ###
    image_logger = build_image_logger(datamodule)
    trainer = Trainer(
        **cfg.trainer,
        logger=[wandb_logger, CSVLogger(save_dir=str(Path.cwd() / "csv_logs"), name="metrics")],
        callbacks=[
            TrainingStatusCallback(),
            image_logger,  # LogPredictionsCallback()
            LearningRateMonitor(),
            ModelCheckpoint(**cfg.checkpoint.callback),
        ],
    )

    trainer.fit(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=str(last_checkpoint) if last_checkpoint is not None else None,
    )
    if getattr(cfg, "test_datamodule", None) is not None:
        # https://github.com/Lightning-AI/lightning/issues/8375#issuecomment-878739663
        torch.distributed.destroy_process_group()
        print(trainer.global_rank)
        trainer = Trainer(**cfg.tester)
        test_datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.test_datamodule, pin_memory=cfg.ngpu != 0
        )
        modelmodule.eval()
        trainer.test(modelmodule, datamodule=test_datamodule)


if __name__ == "__main__":
    main()
