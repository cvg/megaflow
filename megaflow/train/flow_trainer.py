import argparse
import os
import random

import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple

from megaflow.data import datasets
from megaflow.model import MegaFlow
from megaflow.data.datasets import fetch_dataloader
from megaflow.utils.basic import load_ckpt
from megaflow.train.loss import flow_loss_func


class MegaFlowLit(pl.LightningModule):
    """PyTorch Lightning module for MegaFlow optical flow estimation.

    This class implements the training and validation logic for the MegaFlow model
    using PyTorch Lightning framework.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters for the model and training process.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.freeze_bn = getattr(self.args, 'freeze_bn', True)
        self.freeze_backbone = getattr(self.args, 'freeze_backbone', False)
        self.transformer_layers = getattr(self.args, 'transformer_layers', 24)
        self.pretrain_embed = getattr(self.args, 'pretrain_embed', True)

        self.model = MegaFlow(
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            depth=self.transformer_layers,
            freeze_embed=self.pretrain_embed,
            feature_channels=args.feature_channels,
            upsample_factor=args.upsample_factor,
            fix_width=args.fix_width,
            fuse_cnn=args.fuse_cnn,
            use_self_attn_propagation=args.use_self_attn_propagation,
            use_temporal_attn=args.use_temp_attn,
        )
        
        # Load pretrained weight or resume training
        load_ckpt(self, self.args.restore_ckpt, use_dinov3='dinov3' in args.patch_embed, pretrain_embed=self.pretrain_embed)

        # Freeze backbone blocks (if enabled)
        if self.freeze_backbone:
            for name, p in self.named_parameters():
                if "_blocks" in name:
                    p.requires_grad = False

        # Freeze BN affine weights (safe for SyncBN)
        if self.freeze_bn:
            freeze_bn_affine(self)
        
        self.log_kwargs = {"sync_dist": True, "add_dataloader_idx": False, "prog_bar": True}

    def get_random_data_slice(
        self,
        images: torch.Tensor, 
        flow_gts: torch.Tensor, 
        valids: torch.Tensor, 
        min_frames: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes full tensors and returns a random temporal slice 
        based on whether the logic is bidirectional.

        Parameters
        ----------
        images : torch.Tensor
            (B, T_MAX, 3, H, W)
        flow_gts : torch.Tensor
            - (B, 2*(T_MAX-2), 2, H, W) if is_bidirectional=True
            - (B, T_MAX-1, 2, H, W) if is_bidirectional=False
        valids : torch.Tensor
            - (B, 2*(T_MAX-2), H, W) if is_bidirectional=True
            - (B, T_MAX-1, H, W) if is_bidirectional=False
        is_bidirectional : bool
            True: Uses 'centered' logic [BWD, FWD].
            False: Uses 'sequential' logic [FWD only].
        min_frames : int, optional
            Minimum frames for slicing.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (sliced_images, sliced_flow_gts, sliced_valids)
        """
        
        B, T_MAX, C, H, W = images.shape

        if T_MAX == 2:
            return images, flow_gts, valids

        # Bidirectional
        if self.args.bidirectional:
            if min_frames is None:
                min_frames = 3
            if min_frames < 3:
                min_frames = 3 # Centered logic needs at least 3 frames

            T_flows_per_dir_total = flow_gts.shape[1] // 2
            
            n_frames = random.randint(min_frames, T_MAX)
            start_idx = random.randint(0, T_MAX - n_frames)
            end_idx = start_idx + n_frames
            
            sliced_images = images[:, start_idx:end_idx] # (B, n_frames, 3, H, W)
            
            n_flows_per_dir_slice = n_frames - 2
        
            flow_start_idx = start_idx
            flow_end_idx = start_idx + n_flows_per_dir_slice
            
            # Slice the BWD block
            sliced_flow_gts_bwd = flow_gts[:, flow_start_idx:flow_end_idx, ...]
            sliced_valids_bwd = valids[:, flow_start_idx:flow_end_idx, ...]

            # Slice the FWD block
            fwd_start = flow_start_idx + T_flows_per_dir_total
            fwd_end = flow_end_idx + T_flows_per_dir_total
            sliced_flow_gts_fwd = flow_gts[:, fwd_start:fwd_end, ...]
            sliced_valids_fwd = valids[:, fwd_start:fwd_end, ...]
            
            sliced_flow_gts = torch.cat([sliced_flow_gts_bwd, sliced_flow_gts_fwd], dim=1)
            sliced_valids = torch.cat([sliced_valids_bwd, sliced_valids_fwd], dim=1)

            return sliced_images, sliced_flow_gts, sliced_valids

        # Forward-Only
        else:
            if min_frames is None:
                min_frames = 2
            if min_frames < 2:
                min_frames = 2 # Sequential logic needs at least 2 frames

            n_frames = random.randint(min_frames, T_MAX)
            start_idx = random.randint(0, T_MAX - n_frames)
            end_idx = start_idx + n_frames
            
            sliced_images = images[:, start_idx:end_idx] # (B, n_frames, 3, H, W)
            
            flow_start_idx = start_idx
            flow_end_idx = end_idx - 1 
            
            sliced_flow_gts = flow_gts[:, flow_start_idx:flow_end_idx, ...]
            sliced_valids = valids[:, flow_start_idx:flow_end_idx, ...]
            
            return sliced_images, sliced_flow_gts, sliced_valids

    def training_step(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single training step.

        Parameters
        ----------
        data_blob : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flow_gts, valids) tensors.
            - images: Input image sequence of shape (B, 3, 3, H, W)
            - flow_gts: Ground truth flow fields of shape (B, 2, 2, H, W)
            - valids: Validity masks of shape (B, 2, H, W)

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        images, flow_gts, valids = data_blob

        if self.args.random_frame:
            images, flow_gts, valids = self.get_random_data_slice(images, flow_gts, valids)

        if self.args.bidirectional:
            results_dict = self.model.forward_bi(images, num_reg_refine=self.args.iters)
        else:
            results_dict = self.model(images, num_reg_refine=self.args.iters)
        flow_preds = results_dict['flow_preds']
        flow_init = results_dict['flow_init'] if 'flow_init' in results_dict else None
        loss, epe = flow_loss_func(flow_preds, flow_gts, valids, flow_init=flow_init, gamma=self.args.gamma)
        self.log("loss", loss, **self.log_kwargs)
        self.log("epe", epe, **self.log_kwargs)
        return loss

    def validation_step_1(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor], data_name: str, batch_idx: int
    ) -> None:
        """Validation step for chairs, sintel, and spring datasets.

        Parameters
        ----------
        data_blob : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flow_gts, valids) tensors.
        data_name : str
            Name of the validation dataset.
        """
        images, flow_gt, _ = data_blob

        if self.args.bidirectional:
            results_dict = self.model.forward_bi(images, num_reg_refine=self.args.iters)
            flows = results_dict["flow_preds"][-1][:, 1] 
        else:
            results_dict = self.model(images, num_reg_refine=self.args.iters)
            flows = results_dict["flow_preds"][-1]

        epe = torch.sum((flows - flow_gt) ** 2, dim=2).sqrt()

        epe_flat = epe.view(-1)
        
        epe_mean = epe_flat.mean()
        px1 = (epe_flat < 1.0).float().mean()
        px3 = (epe_flat < 3.0).float().mean()
        px5 = (epe_flat < 5.0).float().mean()
        
        self.log(f"val-{data_name}-px1", 100 * (1 - px1), **self.log_kwargs)
        self.log(f"val-{data_name}-px3", 100 * (1 - px3), **self.log_kwargs)
        self.log(f"val-{data_name}-px5", 100 * (1 - px5), **self.log_kwargs)
        self.log(f"val-{data_name}-epe", epe_mean, **self.log_kwargs)

        # --------------------------------------------------

    def validation_step_2(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor], data_name: str, batch_idx: int
    ) -> None:
        """Validation step for KITTI dataset.

        Parameters
        ----------
        data_blob : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flows_gt, valids_gt) tensors.
        data_name : str
            Name of the validation dataset.
        """
        images, flow_gt, valid_gt = data_blob
        if self.args.bidirectional:
            results_dict = self.model.forward_bi(images, num_reg_refine=self.args.iters)
            flows = results_dict["flow_preds"][-1][:,1]
        else:
            results_dict = self.model(images, num_reg_refine=self.args.iters)
            flows = results_dict["flow_preds"][-1]

        epe = torch.sum((flows - flow_gt) ** 2, dim=2).sqrt()
        mag = torch.sum(flow_gt**2, dim=2).sqrt()
        val = (valid_gt >= 0.5)

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        
        epe_list = []
        out_valid_pixels = 0
        num_valid_pixels = 0
        for b in range(out.shape[0]):
            epe_list.append(epe[b][val[b]].mean())
            out_valid_pixels += out[b][val[b]].sum()
            num_valid_pixels += val[b].sum()
            
        epe_mean = torch.stack(epe_list).mean()
        self.log(f"val-{data_name}-epe", epe_mean, **self.log_kwargs)

        # Log Fl-all (F1): Global fraction of outliers
        f1 = 100 * out_valid_pixels / num_valid_pixels
        self.log(f"val-{data_name}-f1", f1, **self.log_kwargs)
        # ---------------------------------------------------

    def validation_step(
        self,
        data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Main validation step that routes to specific validation methods.

        Parameters
        ----------
        data_blob : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing validation data tensors.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the current dataloader, by default 0
        """
        if not self.args.val_datasets:
            return
        data_name = self.args.val_datasets[dataloader_idx]
        if data_name in (
            "chairs",
            "things-clean",
            "things-final",
            "sintel",
            "sintel-clean",
            "sintel-final",
            "spring",
            "spring-1080",
        ):
            self.validation_step_1(data_blob, data_name, batch_idx)
        elif data_name in ("kitti",):
            self.validation_step_2(data_blob, data_name, batch_idx)

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer and scheduler configurations.
        """
        if self.pretrain_embed and self.transformer_layers == 24:
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in self.named_parameters() if '_blocks' in name and param.requires_grad], 'lr': self.args.lr * 0.01},
                {'params': [param for name, param in self.named_parameters() if '_blocks' not in name and param.requires_grad], 'lr': self.args.lr},
                ], weight_decay=self.args.wdecay, eps=self.args.epsilon)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.wdecay,
                eps=self.args.epsilon
            )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.args.lr,
            self.args.num_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="cos", # cosine
            last_epoch=-1,
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
    
def freeze_bn_affine(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MegaFlow training and validation.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters for data loading.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

    def train_dataloader(self) -> data.DataLoader:
        """Get training dataloader.

        Returns
        -------
        data.DataLoader
            Training dataloader instance.
        """
        return fetch_dataloader(self.args)

    def val_dataloader(self) -> list[data.DataLoader]:
        """Get validation dataloaders for different datasets.

        Returns
        -------
        List[data.DataLoader]
            List of validation dataloaders.
        """
        kwargs = {
            "pin_memory": False,
            "shuffle": False,
            "num_workers": self.args.num_workers,
            "drop_last": False,
        }
        val_dataloaders = []
        val_frames = self.args.max_frames # note: fix 2 frame validation
        for val_dataset in self.args.val_datasets:
            if val_dataset == "sintel-clean":
                clean = datasets.n_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "clean"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                loader = data.DataLoader(clean, batch_size=4, **kwargs)
            elif val_dataset == "sintel-final":
                final = datasets.n_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "final"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                loader = data.DataLoader(final, batch_size=4, **kwargs)
            elif val_dataset == "kitti":
                if self.args.bidirectional:
                    kitti = datasets.n_frame_wrapper_val(
                    datasets.KITTI, {"split": "val"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                else: 
                    kitti = datasets.n_frame_wrapper_val(
                        datasets.KITTIN, {"split": "val", }, n_frames=val_frames, bidirectional=self.args.bidirectional, single_flow_only=True
                    )
                loader = data.DataLoader(kitti, batch_size=1, **kwargs)
            elif val_dataset == "spring":
                spring = datasets.n_frame_wrapper_val(
                    datasets.SpringFlowDataset, {"split": "val"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                loader = data.DataLoader(spring, batch_size=4, **kwargs)
            elif val_dataset == "chairs":
                chairs = datasets.n_frame_wrapper_val(
                    datasets.FlyingChairs, {"split": "val"}, n_frames=2 # hard-coded 2 frame for chairs validation
                )
                loader = data.DataLoader(chairs, batch_size=8, **kwargs)
            elif val_dataset == "things-clean":
                things_clean = datasets.n_frame_wrapper_val(
                    datasets.FlyingThings3D, {"split": "val", "dstype": "frames_cleanpass"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                loader = data.DataLoader(things_clean, batch_size=4, **kwargs)
            elif val_dataset == "things-final":
                things_final = datasets.n_frame_wrapper_val(
                    datasets.FlyingThings3D, {"split": "val", "dstype": "frames_finalpass"}, n_frames=val_frames, bidirectional=self.args.bidirectional
                )
                loader = data.DataLoader(things_final, batch_size=4, **kwargs)
            else:
                raise ValueError(f"Unknown validation dataset: {val_dataset}")
            val_dataloaders.append(loader)
        return val_dataloaders
