# Data loading based on https://github.com/aharley/alltracker
import os
import torch
import cv2
import imageio
import numpy as np
import pickle
from pathlib import Path
from megaflow.data.point_utils.pointdataset import PointDataset
from megaflow.utils.data import standardize_test_data
import random
random.seed(42)

class KubricMovifDataset(PointDataset):
    def __init__(
            self,
            data_root='datasets/kubric_au',
            crop_size=(384, 512),
            seq_len=24, 
            traj_per_sample=256,
            traj_max_factor=24, # multiplier on traj_per_sample
            use_augs=True,
            random_seq_len=False,
            random_first_frame=False,
            random_frame_rate=False,
            random_number_traj=False,
            shuffle_frames=False,
            shuffle=True,
            only_first=True,
            # split='training'
    ):
        super(KubricMovifDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
        )


        self.dname = 'kubric%d' % seq_len

        self.only_first = only_first
        self.traj_max_factor = traj_max_factor
        
        self.random_seq_len = random_seq_len
        self.random_first_frame = random_first_frame
        self.random_frame_rate = random_frame_rate
        self.random_number_traj = random_number_traj
        self.shuffle_frames = shuffle_frames
        self.pad_bounds = [10, 100]
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        folder_names = Path(data_root).glob('*/*/')
        folder_names = [str(fn) for fn in folder_names]
        folder_names = sorted(folder_names)
        
        all_seq_names = []
        for fi, fol in enumerate(folder_names):
            npy_path = os.path.join(fol, "annot.npy")
            rgb_path = os.path.join(fol, "frames")
            if os.path.isdir(rgb_path) and os.path.isfile(npy_path):
                img_paths = sorted(os.listdir(rgb_path))
                if len(img_paths)>=seq_len:
                    all_seq_names.append(fol)

        self.seq_names = all_seq_names

        if shuffle:
            random.shuffle(self.seq_names)

    def getitem_helper(self, index):
        gotit = True
        fol = self.seq_names[index]
        npy_path = os.path.join(fol, "annot.npy")
        rgb_path = os.path.join(fol, "frames")

        seq_name = fol.split('/')[-1]

        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []
        for i, img_path in enumerate(img_paths):
            rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))

        rgbs = np.stack(rgbs)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        trajs = annot_dict["point_xys"]
        visibs = annot_dict["point_visibs"]
        valids = annot_dict["point_xys_valid"]
        
        S = len(rgbs)
        # ensure all valid, and then discard valids tensor
        all_valid = np.nonzero(np.sum(valids, axis=0)==S)[0]
        trajs = trajs[:,all_valid]
        visibs = visibs[:,all_valid]

        if self.use_augs and np.random.rand() < 0.5: # time flip
            # time flip
            rgbs = np.flip(rgbs, axis=[0]).copy()
            trajs = np.flip(trajs, axis=[0]).copy()
            visibs = np.flip(visibs, axis=[0]).copy()

        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]
            
        frame_rate = 1
        final_num_traj = self.traj_per_sample
        crop_size = self.crop_size

        # randomize time slice
        min_num_traj = 1
        assert self.traj_per_sample >= min_num_traj
        if self.random_seq_len and self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = int(alpha * 10 + (1 - alpha) * self.seq_len)
            seq_len = np.random.randint(seq_len - 2, seq_len + 2)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        elif self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = 8 * int(alpha * 2 + (1 - alpha) * self.seq_len // 8)
            # seq_len = np.random.randint(seq_len , seq_len + 2)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        elif self.random_seq_len:
            seq_len = np.random.randint(int(self.seq_len / 2), self.seq_len)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        else:
            seq_len = self.seq_len
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        if seq_len < len(rgbs):
            if self.random_first_frame:
                ind0 = np.random.choice(len(rgbs))
                rgb0 = rgbs[ind0]
                traj0 = trajs[ind0]
                visib0 = visibs[ind0]
            
            if seq_len * frame_rate < len(rgbs):
                start_ind = np.random.choice(len(rgbs) - (seq_len * frame_rate), 1)[0]
            else:
                start_ind = 0

            rgbs = rgbs[start_ind : start_ind + seq_len * frame_rate : frame_rate]
            trajs = trajs[start_ind : start_ind + seq_len * frame_rate : frame_rate]
            visibs = visibs[start_ind : start_ind + seq_len * frame_rate : frame_rate]

            if self.random_first_frame:
                rgbs[0] = rgb0
                trajs[0] = traj0
                visibs[0] = visib0
            
        assert seq_len == len(rgbs)

        # ensure no crazy values
        all_valid = np.nonzero(np.sum(np.sum(np.abs(trajs).astype(np.float64), axis=-1)<100000, axis=0)==seq_len)[0]
        trajs = trajs[:,all_valid]
        visibs = visibs[:,all_valid]

        if self.use_augs and np.random.rand() < 0.98:
            rgbs, trajs, visibs = self.add_photometric_augs(rgbs, trajs, visibs, replace=False)
            rgbs, trajs = self.add_spatial_augs(rgbs, trajs, visibs, crop_size)
        else:
            rgbs, trajs = self.crop(rgbs, trajs, crop_size)

        visibs[trajs[:, :, 0] > crop_size[1] - 1] = False
        visibs[trajs[:, :, 0] < 0] = False
        visibs[trajs[:, :, 1] > crop_size[0] - 1] = False
        visibs[trajs[:, :, 1] < 0] = False

        # ensure no crazy values
        all_valid = np.nonzero(np.sum(np.sum(np.abs(trajs).astype(np.float64), axis=-1)<100000, axis=0)==seq_len)[0]
        trajs = trajs[:,all_valid]
        visibs = visibs[:,all_valid]

        if self.shuffle_frames and np.random.rand() < 0.01:
            # shuffle the frames (again)
            perm = np.random.permutation(rgbs.shape[0])
            rgbs = rgbs[perm]
            trajs = trajs[perm]
            visibs = visibs[perm]

        if self.only_first:
            vis_ok = np.nonzero(visibs[0]==1)[0]
            trajs = trajs[:,vis_ok]
            visibs = visibs[:,vis_ok]
        
        visibs = torch.from_numpy(visibs)
        trajs = torch.from_numpy(trajs)

        crop_tensor = torch.tensor(crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1) < 1000.0,
            dim=0,
        )
        trajs = trajs[:, close_pts_inds]
        visibs = visibs[:, close_pts_inds]
        N = trajs.shape[1]

        assert self.only_first
        
        N = trajs.shape[1]
        point_inds = torch.randperm(N)[:self.traj_per_sample*self.traj_max_factor]
        
        if len(point_inds) < self.traj_per_sample:
            gotit = False

        trajs = trajs[:, point_inds]
        visibs = visibs[:, point_inds]
        valids = torch.ones_like(visibs)

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()

        trajs = trajs[:, :self.traj_per_sample*self.traj_max_factor]
        visibs = visibs[:, :self.traj_per_sample*self.traj_max_factor]
        valids = valids[:, :self.traj_per_sample*self.traj_max_factor]
        
        sample = {
            "video": rgbs,
            "trajs": trajs,
            "visibs": visibs,
            "valids": valids,
            "dname": self.dname,
        }
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)
