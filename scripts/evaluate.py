import argparse

import pytorch_lightning as pl
import torch
from config.parser import parse_args
from megaflow.train.flow_trainer import MegaFlowLit, DataModule
from megaflow.train.track_trainer import MegaFlowTrackLit, DataModule as TrackDataModule

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    args = parse_args(parser)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
    )

    import os
    from megaflow.model import MegaFlow

    # Check if we should fallback to HuggingFace
    load_from_hf = False
    if getattr(args, 'restore_ckpt', None) is not None and not os.path.exists(args.restore_ckpt):
        print(f"Warning: Local checkpoint '{args.restore_ckpt}' not found.")
        print("Automatically falling back to HuggingFace pretrained models...")
        args.restore_ckpt = "None"  # Bypass load_ckpt crash
        load_from_hf = True

    if 'kubric' in args.cfg or 'tapvid' in args.cfg:
        datamodule = TrackDataModule(args)
        model = MegaFlowTrackLit(args)
        hf_name = "megaflow-track"
    else:
        datamodule = DataModule(args)
        model = MegaFlowLit(args)
        # Choose the right HF model
        if "zero-shot" in args.cfg:
            hf_name = "megaflow-zero-shot"
        else:
            hf_name = "megaflow-flow"

    if load_from_hf:
        print(f"Downloading and loading '{hf_name}' from HuggingFace...")
        hf_model = MegaFlow.from_pretrained(hf_name)
        model.model.load_state_dict(hf_model.state_dict())
        
    trainer.validate(model, datamodule)
