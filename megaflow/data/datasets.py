# Data loading based on https://github.com/msu-video-group/memfof

import logging
from megaflow.utils.flow_utils import merge_flows
import numpy as np
import torch
import torch.utils.data as data

import os
import json
import random
from glob import glob
import os.path as osp
from megaflow.utils.data import collate_fn_train
from megaflow.utils import frame_utils
from megaflow.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from megaflow.utils.flow_utils import fill_invalid

from functools import reduce
from queue import Queue


def js_read(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


# Placeholder class, may contain:
# - images
# - flows
# - flow_masks
class SceneData:
    def __init__(self):
        self.images = {}
        self.flows = {}
        self.flow_masks = {}
        self.flow_mults = {}
        self.name = ""
        pass


class DataBlob:
    def __init__(self):
        pass


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, scene_params={}):
        self.augmentor = None
        self.sparse = sparse
        self.dataset = "unknown"
        self.subsample_groundtruth = False
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False

        self.scenes = []
        self.scene_params = scene_params

        

    @staticmethod
    def _flow_bfs(flows, s, t, max_depth):
        q = Queue()
        q.put(s)

        prev = {s: None}
        depth = {s: 0}

        ok = False
        while not ok and not q.empty():
            v = q.get()
            if depth[v] > max_depth or v not in flows:
                continue

            for u in flows[v]:
                if u not in prev:
                    prev[u] = v
                    depth[u] = depth[v] + 1
                    q.put(u)

                if u == t:
                    ok = True
                    break

        if not ok:
            return None

        res = []
        while t != s:
            res.append([prev[t], t])
            t = prev[t]

        res.reverse()
        return res

    # Default parameters are for the two frame FW flow case
    def process_scenes(
        self,
        frames=[(0, "left"), (1, "left")],
        flows=[((0, "left"), (1, "left"))],  # (source image, target image)
        sequence_bounds=(0, 1),  # range(min_image + bound[0], max_image + bound[1])
        invalid_images="clip",  # skip or clip
        invalid_flows="merge",
        stride=1,
    ):  # skip or merge
        self.data = []
        for scene in self.scenes:  # `scene` is a SceneData class?
            if not scene.images:
                continue

            min_image = min(scene.images.keys())[0]
            max_image = max(scene.images.keys())[0]

            if sequence_bounds is None:
                i_list = sorted(set((x[0] for x in scene.images.keys())))
            else:
                i_list = list(
                    range(
                        min_image + sequence_bounds[0], max_image + sequence_bounds[1]
                    )
                )
            # Apply stride to ensure non-overlapping windows
            i_list = i_list[::stride]

            for i in i_list:
                valid_data = True

                image_list = []
                for di, cam in frames:
                    image = (i + di, cam)

                    if image not in scene.images:
                        if invalid_images == "skip":
                            valid_data = False
                        elif invalid_images == "clip":
                            image = (max(min_image, min(max_image, image[0])), image[1])
                        else:
                            raise Exception("Invalid image mode")

                    if image not in scene.images:
                        valid_data = False

                    if not valid_data:
                        break
                    else:
                        image_list.append(scene.images[image])

                (
                    flow_list,
                    flow_mask_list,
                    flow_mults_list,
                ) = [], [], []
                for img_pair in flows:
                    if img_pair is None:
                        flow_list.append(None)
                        flow_mask_list.append(None)
                        flow_mults_list.append(None)
                        continue

                    img1_, img2_ = img_pair

                    img1 = (img1_[0] + i, img1_[1])
                    img2 = (img2_[0] + i, img2_[1])

                    img_pairs = []
                    if (img1 not in scene.flows) or (img2 not in scene.flows[img1]):
                        if invalid_flows == "skip":
                            valid_data = False
                        elif invalid_flows == "merge":
                            img_pairs = FlowDataset._flow_bfs(
                                scene.flows,
                                img1,
                                img2,
                                max_depth=abs(img1[0] - img2[0]) + 10,
                            )

                            if img_pairs is None:
                                valid_data = False
                        else:
                            raise Exception("Invalid flow mode")
                    else:
                        img_pairs = [(img1, img2)]

                    if not valid_data:
                        break
                    else:
                        flow_list.append(
                            [scene.flows[img1_][img2_] for img1_, img2_ in img_pairs]
                        )

                    try:
                        flow_mask_list.append(
                            [
                                scene.flow_masks[img1_][img2_]
                                for img1_, img2_ in img_pairs
                            ]
                        )
                    except:
                        flow_mask_list.append(None)

                    try:
                        flow_mults_list.append(
                            [
                                scene.flow_mults[img1_][img2_]
                                for img1_, img2_ in img_pairs
                            ]
                        )
                    except:
                        flow_mults_list.append(None)

                if not valid_data:
                    continue

                new_data = DataBlob()
                new_data.images = image_list
                new_data.flows = flow_list
                new_data.flow_masks = flow_mask_list
                new_data.flow_mults = flow_mults_list
                new_data.extra_info = (scene.name, i, len(self.data), frames, flows)

                self.data.append(new_data)

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)
                raise e

    def fetch(self, index):
        if self.is_test:
            imgs = []
            for image in self.data[index].images:
                img = frame_utils.read_gen(image)
                img = np.array(img).astype(np.uint8)[..., :3]
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                imgs.append(img)

            return torch.stack(imgs), self.data[index].extra_info

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.data)

        imgs = []
        for image in self.data[index].images:
            img = frame_utils.read_gen(image)
            imgs.append(img)

        imgs = [np.array(img).astype(np.uint8) for img in imgs]

        flows, valids = [], []
        for flow_ind in range(len(self.data[index].flows)):
            if self.data[index].flows[flow_ind] is None:
                if len(imgs) > 0:
                    n, m = imgs[0].shape[:2]
                else:
                    n, m = 1

                flows.append(np.zeros((n, m, 2)).astype(np.float32))
                valids.append(np.zeros((n, m)).astype(np.float32))
                continue

            cur_flows, cur_valids = [], []

            if self.sparse:
                if self.dataset == "TartanAir":
                    for f, fm in zip(
                        self.data[index].flows[flow_ind],
                        self.data[index].flow_masks[flow_ind],
                    ):
                        flow = np.load(f)
                        valid = np.load(fm)
                        valid = 1 - valid / 100

                        cur_flows.append(flow)
                        cur_valids.append(valid)
                else:
                    for f in self.data[index].flows[flow_ind]:
                        flow, valid = frame_utils.readFlowKITTI(f)

                        cur_flows.append(flow)
                        cur_valids.append(valid)
            else:
                if self.dataset == "Infinigen":
                    # Inifinigen flow is stored as a 3D numpy array, [Flow, Depth]
                    for f in self.data[index].flows[flow_ind]:
                        flow = np.load(f)
                        flow = flow[..., :2]

                        cur_flows.append(flow)
                elif self.data[index].flow_mults[flow_ind] is not None:
                    for f, f_mult in zip(
                        self.data[index].flows[flow_ind],
                        self.data[index].flow_mults[flow_ind],
                    ):
                        flow = frame_utils.read_gen(f)
                        flow *= np.array(f_mult).astype(flow.dtype)
                        cur_flows.append(flow)
                else:
                    for f in self.data[index].flows[flow_ind]:
                        flow = frame_utils.read_gen(f)
                        cur_flows.append(flow)

            cur_flows = [np.array(flow).astype(np.float32) for flow in cur_flows]
            if not self.sparse:
                cur_valids = [
                    (
                        (np.abs(flow[:, :, 0]) < 1000) & (np.abs(flow[:, :, 1]) < 1000)
                    ).astype(np.float32)
                    for flow in cur_flows
                ]

            if self.subsample_groundtruth:
                # use only every second value in both spatial directions ==> flow will have same dimensions as images
                # used for spring dataset

                cur_flows = [flow[::2, ::2] for flow in cur_flows]
                cur_valids = [valid[::2, ::2] for valid in cur_valids]

            # Merge flows from intermediate steps
            if len(cur_flows) == 1:
                flows.append(cur_flows[0])
                valids.append(cur_valids[0])
            else:
                cur_flow = cur_flows[0]
                cur_valid = cur_valids[0]

                for i in range(1, len(cur_flows)):
                    cur_flow, cur_valid = merge_flows(
                        cur_flow, cur_valid, cur_flows[i], cur_valids[i]
                    )

                cur_flow = fill_invalid(cur_flow, cur_valid)

                flows.append(cur_flow)
                valids.append(cur_valid)

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

        new_flows = []
        for flow in flows:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            flow[torch.isnan(flow)] = 0
            flow[flow.abs() > 1e9] = 0
            new_flows.append(flow)
        flows = new_flows

        if len(valids):
            valids = [torch.from_numpy(valid) for valid in valids]
            if not self.sparse:
                valids = [
                    (valids[i] >= 0.5)
                    & (flows[i][0].abs() < 1000)
                    & (flows[i][1].abs() < 1000)
                    for i in range(len(flows))
                ]
        else:  # should never execute?
            valids = [(flow[0].abs() < 1000) & (flow[1].abs() < 1000) for flow in flows]

        return torch.stack(imgs), torch.stack(flows), torch.stack(valids).float()

    def add_datasets(self, other):
        self.data = self.data + other.data
        del other
        return self

    def __rmul__(self, v):
        self.data = v * self.data
        return self

    def __len__(self):
        return len(self.data)


class MpiSintel(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="train",
        root="../datasets/Sintel",
        dstype="clean",
        scene_params={},
    ):
        super(MpiSintel, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        assert split in ["train", "val", "submission"]
        assert dstype in ["clean", "final"]
        self.dataset = "MpiSintel"

        seq_root = {
            "train": "training",
            "val": "training",
            "submission": "test",
        }[split]

        flow_root = osp.join(root, seq_root, "flow")
        image_root = osp.join(root, seq_root, dstype)

        if split == "submission":
            self.is_test = True

        for scene in sorted(os.listdir(image_root)):

            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            flow_list = sorted(glob(osp.join(flow_root, scene, "*.flo")))

            current_scene = SceneData()
            current_scene.name = scene

            for i in range(len(image_list)):
                current_scene.images[(i, "left")] = image_list[i]

            if split != "submission":
                for i in range(len(flow_list)):
                    current_scene.flows[(i, "left")] = {(i + 1, "left"): flow_list[i]}

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split="train",
        root="../datasets/FlyingChairs_release/data",
        scene_params={},
    ):
        super(FlyingChairs, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        self.dataset = "FlyingChairs"

        images = sorted(glob(osp.join(root, "*.ppm")))
        flows = sorted(glob(osp.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("config/splits/chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "val" and xid == 2
            ):
                current_scene = SceneData()
                current_scene.name = str(i)

                current_scene.images[(0, "left")] = images[2 * i]
                current_scene.images[(1, "left")] = images[2 * i + 1]
                current_scene.flows[(0, "left")] = {(1, "left"): flows[i]}

                self.scenes.append(current_scene)

        self.process_scenes(**scene_params)


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="datasets/FlyingThings3D",
        split="train",
        dstype="frames_cleanpass",
        scene_params={},
        validate_subset=True
    ):
        super(FlyingThings3D, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        self.dataset = "FlyingThings3D"

        seq_root = {
            "train": "TRAIN",
            "val": "TEST",
        }[split]

        image_dirs, flow_dirs, disp_dirs = {}, {}, {}
        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(osp.join(root, dstype, f"{seq_root}/*/*")))
            image_dirs[cam] = sorted([osp.join(f, cam) for f in image_dirs[cam]])

            flow_dirs[cam] = {}
            for direction in ["into_future", "into_past"]:
                flow_dirs[cam][direction] = sorted(
                    glob(osp.join(root, f"optical_flow/{seq_root}/*/*"))
                )
                flow_dirs[cam][direction] = sorted(
                    [osp.join(f, direction, cam) for f in flow_dirs[cam][direction]]
                )

            disp_dirs[cam] = sorted(glob(osp.join(root, f"disparity/{seq_root}/*/*")))
            disp_dirs[cam] = sorted([osp.join(f, cam) for f in disp_dirs[cam]])

        # validate on 1024 subset of test set for fast speed
        if split=='val' and validate_subset:
            num_val_samples = 256
            all_test_samples = len(image_dirs["left"]) + len(image_dirs['right'])

            stride = all_test_samples // num_val_samples
            remove = all_test_samples % num_val_samples

            # uniformly sample a subset
            image_dirs['left'] = image_dirs['left'][remove//2::stride]
            image_dirs['right'] = image_dirs['right'][remove//2::stride]
            flow_dirs['left']['into_future'] = flow_dirs['left']['into_future'][remove//2::stride]
            flow_dirs['right']['into_future'] = flow_dirs['right']['into_future'][remove//2::stride]
            flow_dirs['left']['into_past'] = flow_dirs['left']['into_past'][remove//2::stride]
            flow_dirs['right']['into_past'] = flow_dirs['right']['into_past'][remove//2::stride]
            disp_dirs['left'] = disp_dirs['left'][remove//2::stride]
            disp_dirs['right'] = disp_dirs['right'][remove//2::stride]


        for (
            idir_l,
            idir_r,
            fdir_fw_l,
            fdir_fw_r,
            fdir_bw_l,
            fdir_bw_r,
            ddir_l,
            ddir_r,
        ) in zip(
            image_dirs["left"],
            image_dirs["right"],
            flow_dirs["left"]["into_future"],
            flow_dirs["right"]["into_future"],
            flow_dirs["left"]["into_past"],
            flow_dirs["right"]["into_past"],
            disp_dirs["left"],
            disp_dirs["right"],
        ):
            images_l = sorted(glob(osp.join(idir_l, "*.png")))
            fw_flows_l = sorted(glob(osp.join(fdir_fw_l, "*.pfm")))
            bw_flows_l = sorted(glob(osp.join(fdir_bw_l, "*.pfm")))
            disp_l = sorted(glob(osp.join(ddir_l, "*.pfm")))

            images_r = sorted(glob(osp.join(idir_r, "*.png")))
            fw_flows_r = sorted(glob(osp.join(fdir_fw_r, "*.pfm")))
            bw_flows_r = sorted(glob(osp.join(fdir_bw_r, "*.pfm")))
            disp_r = sorted(glob(osp.join(ddir_r, "*.pfm")))

            current_scene = SceneData()

            for i in range(len(images_l)):
                current_scene.images[(i, "left")] = images_l[i]
                current_scene.flows[(i, "left")] = {
                    (i - 1, "left"): bw_flows_l[i],
                    (i + 1, "left"): fw_flows_l[i],
                }

                current_scene.images[(i, "right")] = images_r[i]
                current_scene.flows[(i, "right")] = {
                    (i - 1, "right"): bw_flows_r[i],
                    (i + 1, "right"): fw_flows_r[i],
                }

                current_scene.flows[(i, "left")][(i, "right")] = disp_l[i]
                current_scene.flows[(i, "right")][(i, "left")] = disp_r[i]

            for k in current_scene.flows:
                current_scene.flow_mults[k] = {}
                for k2 in current_scene.flows[k]:
                    current_scene.flow_mults[k][k2] = 1

            for i in range(len(images_l)):
                current_scene.flow_mults[(i, "right")][(i, "left")] = -1

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)

class KITTIN(FlowDataset):
    def __init__(
        self, aug_params=None, split="train", root="../datasets/KITTI", scene_params={}
    ):
        super(KITTIN, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        assert split in ["train", "val", "test", "submission"]
        self.dataset = "KITTI_N"

        if split == "submission":
            self.is_test = True

        seq_split = {
            "train": "training",
            "val": "training",
            "submission": "testing",
        }[split]

        root = osp.join(root, seq_split)

        image_dir = osp.join(root, "image_2")
        flow_dir = osp.join(root, "flow_occ")
        
        # Get all unique scene IDs (e.g., "000000", "000001", ...)
        scene_files = sorted(glob(osp.join(image_dir, "*_10.png")))
        scene_ids = [f.split('/')[-1][:6] for f in scene_files]

        for scene_id in scene_ids:
                
            current_scene = SceneData()
            current_scene.name = scene_id
            
            image_files = sorted(glob(osp.join(image_dir, f"{scene_id}_*.png")))
            
            for img_file in image_files:
                # Get the frame number ({e.g., 10 for *_10.png)
                frame_idx = int(img_file[-6:-4])
                current_scene.images[(frame_idx, "left")] = img_file
            if split != "submission":
                flow_file = osp.join(flow_dir, f"{scene_id}_10.png")
                current_scene.flows[(10, "left")] = {(11, "left"): flow_file}

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)

        
class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1K", scene_params={}):
        super(HD1K, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        self.dataset = "HD1K"

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            current_scene = SceneData()

            for i in range(len(images)):
                current_scene.images[(i, "left")] = images[i]

            for i in range(len(flows)):
                current_scene.flows[(i, "left")] = {(i + 1, "left"): flows[i]}

            self.scenes.append(current_scene)
            seq_ix += 1

        self.process_scenes(**scene_params)


class SpringFlowDataset(FlowDataset):
    """
    Dataset class for Spring optical flow dataset.
    For train, this dataset returns image1, image2, flow and a data tuple (framenum, scene name, left/right cam, FW/BW direction).
    For test, this dataset returns image1, image2 and a data tuple (framenum, scene name, left/right cam, FW/BW direction).

    root: root directory of the spring dataset (should contain test/train directories)
    split: train/test split
    subsample_groundtruth: If true, return ground truth such that it has the same dimensions as the images (1920x1080px); if false return full 4K resolution
    """

    def __init__(
        self,
        aug_params=None,
        root="datasets/spring",
        split="train",
        subsample_groundtruth=True,
        scene_params={},
    ):
        super(SpringFlowDataset, self).__init__(
            aug_params=aug_params, scene_params=scene_params
        )
        assert split in ["train", "val", "test", "submission"]

        self.dataset = "Spring"

        seq_root = {
            "train": "train",
            "val": "train",
            "test": "train",
            "submission": "test",
        }[split]

        seq_root = os.path.join(root, seq_root)

        if not os.path.exists(seq_root):
            raise ValueError(f"Spring directory does not exist: {seq_root}")

        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == "submission":
            self.is_test = True

        scene_split = js_read(os.path.join("config", "splits", "spring.json"))

        for scene in sorted(os.listdir(seq_root)):
            if scene_split[scene] != split:
                continue

            current_scene = SceneData()
            current_scene.name = scene

            for cam in ["left", "right"]:
                images = sorted(
                    glob(os.path.join(seq_root, scene, f"frame_{cam}", "*.png"))
                )

                for i in range(len(images)):
                    current_scene.images[(i, cam)] = images[i]
                    current_scene.flows[(i, cam)] = {}
                    current_scene.flow_mults[(i, cam)] = {}

                if split != "submission":
                    for direction in ["FW"]:
                        flows = sorted(
                            glob(
                                os.path.join(
                                    seq_root, scene, f"flow_{direction}_{cam}", "*.flo5"
                                )
                            )
                        )

                        for i in range(len(flows)):
                            current_scene.flows[(i, cam)][(i + 1, cam)] = flows[i]
                            current_scene.flow_mults[(i, cam)][(i + 1, cam)] = 1

                    for direction in ["BW"]:
                        flows = sorted(
                            glob(
                                os.path.join(
                                    seq_root, scene, f"flow_{direction}_{cam}", "*.flo5"
                                )
                            )
                        )

                        for i in range(len(flows)):
                            current_scene.flows[(i + 1, cam)][(i, cam)] = flows[i]
                            current_scene.flow_mults[(i + 1, cam)][(i, cam)] = 1

                    if cam == "left":
                        othercam = "right"
                    else:
                        othercam = "left"

                    disps = sorted(
                        glob(os.path.join(seq_root, scene, f"disp1_{cam}", "*.dsp5"))
                    )

                    for i in range(len(disps)):
                        current_scene.flows[(i, cam)][(i, othercam)] = disps[i]
                        current_scene.flow_mults[(i, cam)][(i, othercam)] = (
                            1 if cam == "left" else -1
                        )

            self.scenes.append(current_scene)

        self.process_scenes(**scene_params)

class TartanAir(FlowDataset):
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, aug_params=None, root="datasets/TartanAir", scene_params={}):
        super(TartanAir, self).__init__(
            aug_params=aug_params, scene_params=scene_params, sparse=True
        )
        self.dataset = "TartanAir"
        self.root = root
        self._build_dataset()

        self.process_scenes(**scene_params)

    def _build_dataset(self):
        scenes = glob(osp.join(self.root, "*/*/*"))

        for scene in sorted(scenes):
            current_scene = SceneData()
            current_scene.name = scene

            images = sorted(glob(osp.join(scene, "image_left/*.png")))
            for idx in range(len(images)):
                current_scene.images[(idx, "left")] = images[idx]

            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)

                current_scene.flows[(idx, "left")] = {
                    (idx + 1, "left"): osp.join(
                        scene, "flow", f"{frame0}_{frame1}_flow.npy"
                    )
                }
                current_scene.flow_masks[(idx, "left")] = {
                    (idx + 1, "left"): osp.join(
                        scene, "flow", f"{frame0}_{frame1}_mask.npy"
                    )
                }

            self.scenes.append(current_scene)

def n_frame_wrapper(
    dataset_class, 
    dataset_args, 
    n_frames=2, 
    bidirectional=False,
    add_reversed=True, 
    invalid_images="skip",
    enable_backward=True,
    single_flow_only=False,
):
    """
    Unified wrapper to load N frames.
    
    - bidirectional=True :
        Loads N frames [0..N-1] and 2*(N-2) flows from center frames.
        - bidirectional=True: Loads [BWD, FWD] flows.
        - bidirectional=False: Loads [None, FWD] flows.
        - add_reversed=True: Adds sample [N-1..0] with [FWD, BWD] flows.
        
    - bidirectional=False:
        Loads N frames [0..N-1] and (N-1) forward flows [0->1, 1->2, ...].
        - 'bidirectional' param is ignored.
        - add_reversed=True: Adds sample [N-1..0] with (N-1) backward flows.
    """
    
    if n_frames < 2:
        raise ValueError("n_frames must be 2 or greater.")

    # Frame indices [0, 1, ..., N-1]
    frame_indices = list(range(n_frames))
    
    datasets = []
    for cam in ["left", "right"]:
        
        # Bidirectional
        if bidirectional:
            center_indices = frame_indices[1:-1] # [1, ..., N-2]
            
            gen_frames = [(di, cam) for di in frame_indices]
            gen_flows_forward = []
            gen_flows_backward = []
            
            for di in center_indices:
                prev_frame_idx = di - 1
                next_frame_idx = di + 1
                gen_flows_forward.append( ((di, cam), (next_frame_idx, cam)) )
                
                if enable_backward:
                    gen_flows_backward.append( ((di, cam), (prev_frame_idx, cam)) )
                else:
                    gen_flows_backward.append( None )
            
            # [BWD_BLOCK, FWD_BLOCK]
            gen_flows = gen_flows_backward + gen_flows_forward

            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": gen_frames,
                        "flows": gen_flows,
                        "invalid_images": invalid_images,
                    },
                )
            )

            if add_reversed:
                rev_frames = gen_frames[::-1]
                # Swapped blocks
                rev_flows = gen_flows_forward + gen_flows_backward
                datasets.append(
                    dataset_class(
                        **dataset_args,
                        scene_params={
                            "frames": rev_frames,
                            "flows": rev_flows,
                            "invalid_images": invalid_images,
                        },
                    )
                )

        # Only forward
        else:
            # Pass 1: Forward
            gen_frames_fwd = [(i, cam) for i in frame_indices]
            gen_flows_fwd = []
            if single_flow_only:
                gen_flows_fwd = [ ((0, cam), (1, cam)) ] + [None]*(n_frames - 2)
            else:
                for i in range(n_frames - 1):
                    gen_flows_fwd.append( ((i, cam), (i+1, cam)) ) # (0->1), (1->2)

            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": gen_frames_fwd,
                        "flows": gen_flows_fwd,
                        "invalid_images": invalid_images,
                    },
                )
            )

            # Pass 2: Backward
            if add_reversed:
                gen_frames_bwd = gen_frames_fwd[::-1] # [N-1, ..., 0]
                gen_flows_bwd = []
                if single_flow_only:
                    gen_flows_bwd = [ ((1, cam), (0, cam)) ] + [None]*(n_frames - 2)
                else:
                    for i in range(n_frames - 1):
                        src_idx = frame_indices[n_frames - 1 - i]
                        tgt_idx = frame_indices[n_frames - 2 - i]
                        gen_flows_bwd.append( ((src_idx, cam), (tgt_idx, cam)) ) # (N-1 -> N-2), ...

                datasets.append(
                    dataset_class(
                        **dataset_args,
                        scene_params={
                            "frames": gen_frames_bwd,
                            "flows": gen_flows_bwd,
                            "invalid_images": invalid_images,
                        },
                    )
                )
        
    if not datasets:
        return None
    
    return reduce(lambda x, y: x.add_datasets(y), datasets)

def n_frame_wrapper_val(
    dataset_class, 
    dataset_args, 
    n_frames=2, 
    bidirectional=False,
    single_flow_only=False,
):
    
    if n_frames < 2:
        raise ValueError("n_frames must be 2 or greater.")
    
    # Calculate exact stride for non-overlapping multi-frame chunks
    window_stride = 1 if single_flow_only else (n_frames - 1)

    datasets = []
    for cam in ["left", "right"]:

        if bidirectional:
            start_idx = -((n_frames - 1) // 2)
            end_idx = n_frames // 2
            frame_indices = list(range(start_idx, end_idx + 1))
            center_indices = frame_indices[1:-1]

            gen_frames = [(di, cam) for di in frame_indices]
            rev_frames = gen_frames[::-1]
            
            gen_flows_forward = []
            gen_flows_backward = []
            
            # 3. Generate the flow lists
            for di in center_indices:
                prev_frame_idx = di - 1
                next_frame_idx = di + 1
                # (di -> di+1)
                gen_flows_forward.append( ((di, cam), (next_frame_idx, cam)) )
                # (di -> di-1)
                gen_flows_backward.append( ((di, cam), (prev_frame_idx, cam)) )

            # 4. Add the dataset for forward flows
            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": gen_frames,
                        "flows": gen_flows_forward,
                    },
                )
            )
            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": rev_frames,
                        "flows": gen_flows_backward,
                    },
                )
            )
        else:
            frame_indices = list(range(n_frames)) # [0, 1, ..., N-1]
            gen_frames_fwd = [(i, cam) for i in frame_indices]
            gen_flows_fwd = []
            if single_flow_only:
                gen_flows_fwd = [ ((0, cam), (1, cam)) ] + [None]*(n_frames - 2)
            else:
                for i in range(n_frames - 1):
                    gen_flows_fwd.append( ((i, cam), (i+1, cam)) ) # (0->1), (1->2)

            datasets.append(
                dataset_class(
                    **dataset_args,
                    scene_params={
                        "frames": gen_frames_fwd,
                        "flows": gen_flows_fwd,
                        "stride": window_stride,
                    },
                )
            )
        
    return reduce(lambda x, y: x.add_datasets(y), datasets)

def fetch_dataloader(args):
    """Create the data loader for the corresponding training set"""

    if args.dataset == 'chairs':
        assert args.bidirectional == False, "Chairs dataset only support forward only flow."
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}

        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.dataset == "things":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.4,
            "max_scale": +0.8,
            "do_flip": True,
        }

        clean_dataset = n_frame_wrapper(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_cleanpass"}, n_frames=args.max_frames, bidirectional=args.bidirectional
        )
        final_dataset = n_frame_wrapper(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_finalpass"}, n_frames=args.max_frames, bidirectional=args.bidirectional
        )

        train_dataset = clean_dataset + final_dataset

    elif args.dataset == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": +0.4,
            "do_flip": False,
        }
        train_dataset = n_frame_wrapper(
            KITTIN, {"aug_params": aug_params, "split": "train"}, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional, invalid_images="skip"
        )

    elif args.dataset == "sintel":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": +0.6,
            "do_flip": True,
        }
        sintel_clean = n_frame_wrapper(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": 'train',
                        "dstype": "clean",
                    }, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional
                )
        
        sintel_final = n_frame_wrapper(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": 'train',
                        "dstype": "final",
                    }, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional
                )
        train_dataset = sintel_clean + sintel_final

    elif args.dataset == "spring":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": 0.0,
            "max_scale": +0.2,
            "do_flip": True,
        }
        train_dataset = n_frame_wrapper(
            SpringFlowDataset, {"aug_params": aug_params, "subsample_groundtruth": True}, n_frames=args.max_frames, bidirectional=args.bidirectional
        )

    elif args.dataset == "spring-full":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": 0.0,
            "max_scale": +0.2,
            "do_flip": True,
        }
        train_dataset = reduce(
            lambda x, y: x.add_datasets(y),
            [
                n_frame_wrapper(
                    SpringFlowDataset,
                    {
                        "split": cur_sp,
                        "aug_params": aug_params,
                        "subsample_groundtruth": True,
                    }, n_frames=args.max_frames,
                )
                for cur_sp in ["train", "val", "test"]
            ],
        )

    elif args.dataset == "TartanAir":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": +0.4,
            "do_flip": True,
        }
        train_dataset = n_frame_wrapper(TartanAir, {"aug_params": aug_params}, n_frames=args.max_frames, bidirectional=args.bidirectional)

    elif args.dataset == "TSKH":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": +0.6,
            "do_flip": True,
        }

        things_clean = n_frame_wrapper(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_cleanpass"}, n_frames=args.max_frames, bidirectional=args.bidirectional
        )
        things_final = n_frame_wrapper(
            FlyingThings3D, {"aug_params": aug_params, "dstype": "frames_finalpass"}, n_frames=args.max_frames, bidirectional=args.bidirectional
        )
        things = things_clean + things_final

        sintel_clean = n_frame_wrapper(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": 'train',
                        "dstype": "clean",
                    }, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional
                )
        sintel_final = n_frame_wrapper(
                    MpiSintel,
                    {
                        "aug_params": aug_params,
                        "split": 'train',
                        "dstype": "final",
                    }, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional
                )

        kitti = n_frame_wrapper(
                KITTIN, {
                    "aug_params": {
                        "crop_size": args.image_size,
                        "min_scale": -0.3,
                        "max_scale": +0.5,
                        "do_flip": True,
                    }, "split": "train"}, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional, single_flow_only=True
            )

        hd1k = n_frame_wrapper(
            HD1K,
            {
                "aug_params": {
                    "crop_size": args.image_size,
                    "min_scale": -0.5,
                    "max_scale": +0.2,
                    "do_flip": True,
                }
            }, n_frames=args.max_frames, enable_backward=False, bidirectional=args.bidirectional
        )

        train_dataset = (
            80 * sintel_clean + 80 * sintel_final + 80 * hd1k + 320 * kitti + things
        )

    elif args.dataset == 'kubric':
        from megaflow.data.point_utils.kubric_movif_dataset import KubricMovifDataset
        from torch.utils.data import ConcatDataset

        short_len, long_len = 16, 32
        train_dataset_short = KubricMovifDataset(data_root='datasets/kubric_au', crop_size=(384, 512), seq_len=short_len)
        train_dataset_long = KubricMovifDataset(data_root='datasets/kubric_long', crop_size=(256, 384), seq_len=long_len, random_first_frame=False)
        train_dataset = ConcatDataset([train_dataset_short, train_dataset_long])

    else:
        raise ValueError(f"Invalid dataset name {args.dataset}")

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn_train if args.dataset == 'kubric' else None
    )

    logging.info("Training with %d image pairs" % len(train_dataset))
    return train_loader

    