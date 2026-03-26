import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import TorchSyncBatchNorm
from pytorch_lightning import seed_everything

from config.parser import parse_args
from megaflow.train.flow_trainer import MegaFlowLit, DataModule

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

def detect_cluster(args: argparse.Namespace) -> argparse.Namespace:
    if all(env in os.environ for env in ("SLURM_NTASKS_PER_NODE", "SLURM_JOB_NUM_NODES")):
        args.devices = int(os.environ["SLURM_NTASKS_PER_NODE"])
        args.num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    elif all(env in os.environ for env in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        args.devices = int(os.environ["LOCAL_WORLD_SIZE"])
        args.num_nodes = int(os.environ["WORLD_SIZE"]) // args.devices
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="experiment config file name", required=True)
    args = parse_args(parser)
    args = detect_cluster(args)

    if args.effective_batch_size % (args.num_nodes * args.devices * args.accumulate_grad_batches) != 0:
        raise ValueError(
            f"Requested effective_batch_size={args.effective_batch_size} can not be split into "
            f"{args.num_nodes} nodes with {args.devices} devices each with "
            f"accumulate_grad_batches={args.accumulate_grad_batches}."
        )

    args.batch_size = int(args.effective_batch_size / (args.num_nodes * args.devices * args.accumulate_grad_batches))

    monitor = LearningRateMonitor()
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename=args.name,
        monitor=args.monitor,
        every_n_train_steps=args.save_steps if args.monitor is None else None,
    )

    logger = WandbLogger(
        project="MegaFlow",
        config=vars(args),
        log_model=False,
        name=args.name,
        version=args.name,
        id=args.name,
        resume="auto"
    )

    plugins = [
        TorchSyncBatchNorm(),
    ]
    seed_everything(42, workers=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=args.num_nodes,
        logger=logger,
        gradient_clip_val=args.clip,
        precision="bf16-mixed",
        max_steps=args.num_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_steps * args.accumulate_grad_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[monitor, checkpoint],
        plugins=plugins,
        enable_progress_bar=False
    )

    model = MegaFlowLit(args)
    datamodule = DataModule(args)
    trainer.fit(model, datamodule)
