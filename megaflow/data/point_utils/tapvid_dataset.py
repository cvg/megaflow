# Data loading based on https://github.com/aharley/alltracker
import os
import torch
import cv2, io
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
from megaflow.data.point_utils.pointdataset import PointDataset
from megaflow.utils.data import standardize_test_data
import random
random.seed(42)

class DavisDataset(PointDataset):
    def __init__(
            self,
            data_root='datasets/TAP_Vid/tapvid_davis',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(DavisDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        self.dname = 'davis'
        self.only_first = only_first
        
        input_path = '%s/tapvid_davis.pkl' % data_root
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data

        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        visibs = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when not occluded
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = visibs.copy()
        
        rgbs, trajs, visibs, valids = standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
        
        if self.crop_size is not None:
            rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        # in this data, 1.0,1.0 should lie at the bottom-right corner pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W
        trajs[:,:,1] *= H

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        valids = torch.from_numpy(valids).float() # S,N
        visibs = torch.from_numpy(visibs).float() # S,N

        sample = {
            "video": rgbs,
            "trajs": trajs,
            "visibs": visibs,
            "valids": valids,
            "dname": self.dname,
        }

        return sample, True

    def __len__(self):
        return len(self.data)
    



def decode(frame):
    byteio = io.BytesIO(frame)
    img = Image.open(byteio)
    return np.array(img)

class KineticsDataset(PointDataset):
    def __init__(
        self,
        data_root='datasets/TAP_Vid/tapvid_kinetics',
        crop_size=(384,512),
        seq_len=None,
        only_first=False,
    ):
        super(KineticsDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )



        self.dname = 'kinetics'
        self.only_first = only_first
        
        self.data = []
        for vid_pkl in sorted(list(Path(data_root).glob('*.pkl')))[:]:
            vid_pkl = vid_pkl.name

            input_path = "%s/%s" % (data_root, vid_pkl)
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            self.data += data

        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        if isinstance(rgbs[0], bytes):  # decode if needed
            rgbs = [decode(frame) for frame in rgbs]
        trajs = dat['points'] # N,S,2 array
        visibs = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when visib
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = visibs.copy()

        rgbs, trajs, visibs, valids = standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)

        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1

        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        visibs = torch.from_numpy(visibs).float() # S,N
        valids = torch.from_numpy(valids).float() # S,N

        sample = {
            "video": rgbs,
            "trajs": trajs,
            "visibs": visibs,
            "valids": valids,
            "dname": self.dname,
        }
        return sample, True

    def __len__(self):
        return len(self.data)
    
class RGBStackingDataset(PointDataset):
    def __init__(
            self,
            data_root='datasets/TAP_Vid/tapvid_rgb_stacking',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(RGBStackingDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )



        self.dname = 'rgbstacking'
        self.only_first = only_first
        
        input_path = '%s/tapvid_rgb_stacking.pkl' % data_root
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())
        self.data = data

        
    def __getitem__(self, index):
        dat = self.data[index]
        rgbs = dat['video'] # list of H,W,C uint8 images
        trajs = dat['points'] # N,S,2 array
        visibs = 1-dat['occluded'] # N,S array
        # note the annotations are only valid when visib
        valids = visibs.copy()
        
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = valids.transpose(1,0) # S,N

        rgbs, trajs, visibs, valids = standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
        
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1
        
        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        visibs = torch.from_numpy(visibs).float() # S,N
        valids = torch.from_numpy(valids).float() # S,N

        sample = {
            "video": rgbs,
            "trajs": trajs,
            "visibs": visibs,
            "valids": valids,
            "dname": self.dname,
        }

        return sample, True

    def __len__(self):
        return len(self.data)
    



class RobotapDataset(PointDataset):
    def __init__(
            self,
            data_root='datasets/TAP_Vid/robotap',
            crop_size=(384,512),
            seq_len=None,
            only_first=False,
    ):
        super(RobotapDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
        )

        self.dname = 'robo'
        self.only_first = only_first
        
        self.val_pkls = ['robotap_split3.pkl', 'robotap_split4.pkl']

        self.data = []
        for vid_pkl in self.val_pkls:
            input_path = "%s/%s" % (data_root, vid_pkl)
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            keys = list(data.keys())
            self.data += [data[key] for key in keys]
        # print("found %d videos in %s" % (len(self.data), data_root))

    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        dat = self.data[index]
        rgbs = dat["video"]  # list of H,W,C uint8 images
        trajs = dat["points"]  # N,S,2 array
        visibs = 1 - dat["occluded"]  # N,S array

        # note the annotations are only valid when not occluded
        trajs = trajs.transpose(1,0,2) # S,N,2
        visibs = visibs.transpose(1,0) # S,N
        valids = visibs.copy()

        rgbs, trajs, visibs, valids = standardize_test_data(
            rgbs, trajs, visibs, valids, only_first=self.only_first, seq_len=self.seq_len)
        
        rgbs = [cv2.resize(rgb, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W = rgbs[0].shape[:2]
        trajs[:,:,0] *= W-1
        trajs[:,:,1] *= H-1
        
        rgbs = torch.from_numpy(np.stack(rgbs,0)).permute(0,3,1,2).contiguous().float() # S,C,H,W
        trajs = torch.from_numpy(trajs).float() # S,N,2
        visibs = torch.from_numpy(visibs).float() # S,N
        valids = torch.from_numpy(valids).float() # S,N

        if self.seq_len is not None:
            rgbs = rgbs[:self.seq_len]
            trajs = trajs[:self.seq_len]
            valids = valids[:self.seq_len]
            visibs = visibs[:self.seq_len]

        sample = {
            "video": rgbs,
            "trajs": trajs,
            "visibs": visibs,
            "valids": valids,
            "dname": self.dname,
        }

        return sample, True