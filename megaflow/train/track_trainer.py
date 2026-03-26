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

from megaflow.data.point_utils.tapvid_dataset import DavisDataset, RGBStackingDataset, KineticsDataset, RobotapDataset
from megaflow.data.point_utils.kubric_movif_dataset import KubricMovifDataset
from megaflow.model import MegaFlow
from megaflow.data.datasets import fetch_dataloader
from megaflow.utils.basic import gridcloud2d, load_ckpt
from megaflow.train.loss import pointflow_loss_func


class MegaFlowTrackLit(pl.LightningModule):
    """PyTorch Lightning module for MegaFlow point tracking estimation.

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

        self.model = MegaFlow(
            patch_embed=args.patch_embed,
            patch_size=args.patch_size,
            feature_channels=args.feature_channels,
            upsample_factor=args.upsample_factor,
            fix_width=args.fix_width,
            fuse_cnn=args.fuse_cnn,
            use_self_attn_propagation=args.use_self_attn_propagation,
            use_temporal_attn=args.use_temp_attn,
            seq_len=args.seq_len,
        )
        
        # Load pretrained weight or resume training
        load_ckpt(self, self.args.restore_ckpt, use_dinov3='dinov3' in args.patch_embed)

        # Freeze backbone blocks (if enabled)
        if self.freeze_backbone:
            for name, p in self.named_parameters():
                if "_blocks" in name:
                    p.requires_grad = False

        # Freeze BN affine weights (safe for SyncBN)
        if self.freeze_bn:
            freeze_bn_affine(self)
        
        self.log_kwargs = {"sync_dist": True, "add_dataloader_idx": False, "prog_bar": True}


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
        batch, gotit = data_blob
        
        if not all(gotit):
            return None

        rgbs = batch["video"]
        trajs_g = batch["trajs"]
        vis_g = batch["visibs"]
        valids = batch["valids"]

        if trajs_g.shape[-2]==0 or torch.any(vis_g[:,0].reshape(-1)==0):

            return None

        results_dict = self.model.forward_track(rgbs, num_reg_refine=self.args.iters) #-1 if T > 24 and not args.max_len else args.num_reg_refine)

        flow_preds = results_dict['flow_preds']
        all_flows_e = results_dict['flow_final'] if 'flow_final' in results_dict else None
        loss, metrics = pointflow_loss_func(flow_preds, trajs_g, valids, vis_g, all_flows_e, seq_len=self.args.seq_len, gamma=self.args.gamma)
        self.log("loss", loss, **self.log_kwargs)
        self.log_dict(metrics, **self.log_kwargs)
        return loss

    def validation_step(
        self,
        data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if getattr(self.args, 'val_datasets', None) is None:
            return
            
        data_name = self.args.val_datasets[dataloader_idx]
        thrs = [1, 2, 4, 8, 16]
        
        batch, gotit = data_blob
        if not all(gotit):
            return None

        rgbs = batch["video"]
        trajs_g = batch["trajs"]
        vis_g = batch["visibs"]
        valids_g = batch["valids"]

        _, T, C, H, W = rgbs.shape
        B, T1, N, D = trajs_g.shape
        assert T == T1, f"Mismatch in time dimension between rgbs ({T}) and trajs_g ({T1})"
        
        _, first_positive_inds = torch.max(vis_g.float(), dim=1)
        query_points_all = []
        grid_xy = gridcloud2d(1, H, W, norm=False, device=self.device).float() # 1,H*W,2
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W
        trajs_e = torch.zeros([B, T, N, 2], device=self.device)
        
        for first_positive_ind in torch.unique(first_positive_inds):
            chunk_pt_idxs = torch.nonzero(first_positive_inds[0]==first_positive_ind, as_tuple=False)[:, 0]  # K
            chunk_pts = trajs_g[:, first_positive_ind[None].repeat(chunk_pt_idxs.shape[0]), chunk_pt_idxs]  # B, K, 2
            query_points_all.append(torch.cat([first_positive_inds[:, chunk_pt_idxs, None], chunk_pts], dim=2))
            
            traj_maps_e = grid_xy.repeat(1,T,1,1,1) # B,T,2,H,W
            if first_positive_ind < T-1:
                rgbs_ = rgbs[:, first_positive_ind:]
                # if T > 512:
                results_dict = self.model.forward_track_sliding(rgbs_, num_reg_refine=self.args.iters)
                # else:
                    # results_dict = self.model.forward_track(rgbs_, num_reg_refine=self.args.iters)

                flow = results_dict['flow_final'].to(device=self.device)
                
                forward_traj_maps_e = flow.cuda() + grid_xy # B,Tf,2,H,W
                traj_maps_e[:,first_positive_ind:] = forward_traj_maps_e
                
            xyt = trajs_g[:,first_positive_ind].round().long()[0, chunk_pt_idxs] # K,2
            trajs_e_chunk = traj_maps_e[:, :, :, xyt[:,1], xyt[:,0]] # B,T,2,K
            trajs_e_chunk = trajs_e_chunk.permute(0,1,3,2) # B,T,K,2
            trajs_e.scatter_add_(2, chunk_pt_idxs[None, None, :, None].repeat(1, trajs_e_chunk.shape[1], 1, 2), trajs_e_chunk)
        
        t_indices = torch.arange(T, device=self.device).view(1, T, 1) # [1, T, 1]
        query_frame = first_positive_inds.unsqueeze(1)                # [B, 1, N]
        evaluation_points = t_indices > query_frame                   # [B, T, N]
        
        visible = vis_g.bool() # [B, T, N]
        
        # Convert coordinates and compute threshold limits
        sc_pt = torch.tensor([(W - 1) / 255.0, (H - 1) / 255.0], device=self.device).view(1, 1, 1, 2)
        dist_sq = torch.sum(torch.square((trajs_e - trajs_g) / sc_pt), dim=-1) # [B, T, N]

        # The denominator counts points that are visible AND in the evaluation window
        valid_eval_mask = visible & evaluation_points 
        count_visible_points = valid_eval_mask.sum(dim=(1, 2)) # Shape: [B]
        
        step_metrics = {}
        d_sum = 0.0
        
        for thr in thrs:
            within_dist = dist_sq < (thr ** 2)
            is_correct = within_dist & visible
            
            # Numerator: Points that are correct AND in the evaluation window
            count_correct = (is_correct & evaluation_points).sum(dim=(1, 2)) # Shape: [B]
            
            frac_correct = count_correct.float() / count_visible_points.clamp(min=1.0).float()
            
            d_thr_val = frac_correct.mean().item() * 100.0
            d_sum += d_thr_val
            step_metrics[f"val-{data_name}-d_{thr}"] = d_thr_val
            
        step_metrics[f"val-{data_name}-d_avg"] = d_sum / len(thrs)

        epe = torch.sum((trajs_e - trajs_g) ** 2, dim=-1).sqrt()
        val_mask = (valids_g * vis_g).bool()
        
        if val_mask.any():
            epe_mean = epe[val_mask].mean().item()
            step_metrics[f"val-{data_name}-epe"] = epe_mean
            
        self.log_dict(step_metrics, **self.log_kwargs)
        

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW([
            {'params': [param for name, param in self.named_parameters() if '_blocks' in name and param.requires_grad], 'lr': self.args.lr * 0.01},
            {'params': [param for name, param in self.named_parameters() if '_blocks' not in name and param.requires_grad], 'lr': self.args.lr},
            ], weight_decay=self.args.wdecay, eps=self.args.epsilon)

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
    """PyTorch Lightning DataModule for Megaflow training and validation.

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
        for val_dataset in self.args.val_datasets:
            if val_dataset == 'kubric':
                val_dataset = KubricMovifDataset(seq_len=24, crop_size=self.args.image_size, only_first=False, use_augs=False)
                loader = data.DataLoader(val_dataset, batch_size=1, **kwargs)
            elif val_dataset == 'davis':
                val_dataset = DavisDataset(only_first=False, crop_size=self.args.image_size)
                loader = data.DataLoader(val_dataset, batch_size=1, **kwargs)
            elif val_dataset == 'rgbstack':
                val_dataset = RGBStackingDataset(only_first=False, crop_size=self.args.image_size)
                loader = data.DataLoader(val_dataset, batch_size=1, **kwargs)
            elif val_dataset == 'kinetics':
                val_dataset = KineticsDataset(only_first=True, crop_size=self.args.image_size)
                loader = data.DataLoader(val_dataset, batch_size=1, **kwargs)
            elif val_dataset == 'robotap':
                val_dataset = RobotapDataset(only_first=True, crop_size=self.args.image_size)
                loader = data.DataLoader(val_dataset, batch_size=1, **kwargs)
            val_dataloaders.append(loader)
        return val_dataloaders
