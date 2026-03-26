import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np

def replace_invalid_xys_with_nearest(xys, valids):
    # replace invalid xys with nearby ones
    invalid_idx = np.where(valids==0)[0]
    valid_idx = np.where(valids==1)[0]
    for idx in invalid_idx:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
        xys[idx] = xys[nearest]
    return xys

def standardize_test_data(rgbs, trajs, visibs, valids, S_cap=600, only_first=False, seq_len=None):
    trajs = trajs.astype(np.float32) # S,N,2
    visibs = visibs.astype(np.float32) # S,N
    valids = valids.astype(np.float32) # S,N

    # only take tracks that make sense
    visval_ok = np.sum(valids*visibs, axis=0) > 1
    trajs = trajs[:,visval_ok]
    visibs = visibs[:,visval_ok]
    valids = valids[:,visval_ok]

    # fill in missing data (for visualization)
    N = trajs.shape[1]
    for ni in range(N):
        trajs[:,ni] = replace_invalid_xys_with_nearest(trajs[:,ni], valids[:,ni])

    # use S_cap or seq_len (legacy)
    if seq_len is not None:
        S = min(len(rgbs), seq_len)
    else:
        S = len(rgbs)
    S = min(S, S_cap)

    if only_first:
        # we'll find the best frame to start on
        best_count = 0
        best_ind = 0

        for si in range(0,len(rgbs)-64):
            # try this slice
            visibs_ = visibs[si:min(si+S,len(rgbs)+1)] # S,N
            valids_ = valids[si:min(si+S,len(rgbs)+1)] # S,N
            visval_ok0 = (visibs_[0]*valids_[0]) > 0 # N
            visval_okA = np.sum(visibs_*valids_, axis=0) > 1 # N
            all_ok = visval_ok0 & visval_okA

            if np.sum(all_ok) > best_count:
                best_count = np.sum(all_ok)
                best_ind = si
        si = best_ind
        rgbs = rgbs[si:si+S]
        trajs = trajs[si:si+S]
        visibs = visibs[si:si+S]
        valids = valids[si:si+S]
        vis_ok0 = visibs[0] > 0  # N
        trajs = trajs[:,vis_ok0]
        visibs = visibs[:,vis_ok0]
        valids = valids[:,vis_ok0]


    if seq_len is not None:
        rgbs = rgbs[:seq_len]
        trajs = trajs[:seq_len]
        valids = valids[:seq_len]
        visibs = visibs[:seq_len]
    
    # req two timesteps valid (after seqlen trim)
    visval_ok = np.sum(visibs*valids, axis=0) > 1
    trajs = trajs[:,visval_ok]
    valids = valids[:,visval_ok]
    visibs = visibs[:,visval_ok]

    return rgbs, trajs, visibs, valids

def collate_fn_train(batch):
    """
    Collate function for video tracks data.
    Returns a batch dictionary and a list of flags.
    """
    # batch is a list of (sample_dict, gotit_flag)
    gotit = [item[1] for item in batch]
    
    # Extract and stack tensors from the dictionaries
    video = torch.stack([item[0]["video"] for item in batch], dim=0)
    trajs = torch.stack([item[0]["trajs"] for item in batch], dim=0)
    visibs = torch.stack([item[0]["visibs"] for item in batch], dim=0)
    valids = torch.stack([item[0]["valids"] for item in batch], dim=0)
    
    # Collect non-tensor data as a list
    dnames = [item[0]["dname"] for item in batch]

    batch_dict = {
        "video": video,
        "trajs": trajs,
        "visibs": visibs,
        "valids": valids,
        "dname": dnames,
    }

    return batch_dict, gotit


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
