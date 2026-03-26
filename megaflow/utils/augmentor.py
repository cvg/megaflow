import numpy as np
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision.transforms import ColorJitter


class FlowAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        do_flip=True,
        pwc_aug=False,
        do_rotate=False,
        pre_scale=0.0,
    ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.pre_scale = pre_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.do_rotate = do_rotate
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.rotate_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        self.pwc_aug = pwc_aug
        if self.pwc_aug:
            print("[Using pwc-style spatial augmentation]")

    def color_transform(self, imgs):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            for i in range(len(imgs)):
                imgs[i] = np.array(
                    self.photo_aug(Image.fromarray(imgs[i])), dtype=np.uint8
                )
        # symmetric
        else:
            cnt_imgs = len(imgs)
            image_stack = np.concatenate(imgs, axis=0)
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            imgs = list(np.split(image_stack, cnt_imgs, axis=0))

        return imgs

    def eraser_transform(self, imgs, bounds=[50, 100]):
        """Occlusion augmentation (only for last frame)"""

        img_cnt = len(imgs)
        ht, wd, _ = imgs[0].shape

        if img_cnt >= 3:
            img_inds = [0, img_cnt - 1]
        else:
            img_inds = [img_cnt - 1]

        if np.random.rand() < self.eraser_aug_prob:
            for img_ind in img_inds:
                mean_color = np.mean(imgs[img_ind].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    imgs[img_ind][y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return imgs

    def spatial_transform(self, imgs, flows, valids):
        if self.pre_scale != 0:
            cur_scale = 2**self.pre_scale

            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            for i in range(len(flows)):
                flows[i] = cv2.resize(
                    flows[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                flows[i] = flows[i] * [cur_scale, cur_scale]

            for i in range(len(valids)):
                valids[i] = cv2.resize(
                    valids[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                valids[i] = valids[i].round()

        if self.do_rotate:
            if np.random.rand() < self.rotate_prob:
                if np.random.rand() < 0.5:
                    num_rot = 1
                else:
                    num_rot = -1

                for i in range(len(imgs)):
                    imgs[i] = np.rot90(imgs[i], k=num_rot)

                for i in range(len(flows)):
                    if num_rot == 1:
                        flows[i] = np.rot90(flows[i], k=1)[:, :, ::-1] * [-1.0, 1.0]
                    elif num_rot == -1:
                        flows[i] = np.rot90(flows[i], k=-1)[:, :, ::-1] * [1.0, -1.0]

                for i in range(len(valids)):
                    valids[i] = np.rot90(valids[i], k=num_rot)

            pad_t = 0
            pad_b = 0
            pad_l = 0
            pad_r = 0
            if self.crop_size[0] > imgs[0].shape[0]:
                pad_b = self.crop_size[0] - imgs[0].shape[0]
            if self.crop_size[1] > imgs[0].shape[1]:
                pad_r = self.crop_size[1] - imgs[0].shape[1]
            if pad_b != 0 or pad_r != 0:
                for i in range(len(imgs)):
                    imgs[i] = np.pad(
                        imgs[i],
                        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                        "constant",
                        constant_values=((0, 0), (0, 0), (0, 0)),
                    )

                for i in range(len(flows)):
                    flows[i] = np.pad(
                        flows[i],
                        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                        "constant",
                        constant_values=((0, 0), (0, 0), (0, 0)),
                    )

                for i in range(len(valids)):
                    valids[i] = np.pad(
                        valids[i],
                        ((pad_t, pad_b), (pad_l, pad_r)),
                        "constant",
                        constant_values=((0, 0), (0, 0)),
                    )

        # randomly sample scale
        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i],
                    None,
                    fx=scale_x,
                    fy=scale_y,
                    interpolation=cv2.INTER_LINEAR,
                )

            for i in range(len(flows)):
                flows[i] = cv2.resize(
                    flows[i],
                    None,
                    fx=scale_x,
                    fy=scale_y,
                    interpolation=cv2.INTER_LINEAR,
                )
                flows[i] = flows[i] * [scale_x, scale_y]

            for i in range(len(valids)):
                valids[i] = cv2.resize(
                    valids[i],
                    None,
                    fx=scale_x,
                    fy=scale_y,
                    interpolation=cv2.INTER_LINEAR,
                )
                valids[i] = valids[i].round()

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                for i in range(len(imgs)):
                    imgs[i] = imgs[i][:, ::-1]

                for i in range(len(flows)):
                    flows[i] = flows[i][:, ::-1] * [-1.0, 1.0]

                for i in range(len(valids)):
                    valids[i] = valids[i][:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                for i in range(len(imgs)):
                    imgs[i] = imgs[i][::-1, :]

                for i in range(len(flows)):
                    flows[i] = flows[i][::-1, :] * [1.0, -1.0]

                for i in range(len(valids)):
                    valids[i] = valids[i][::-1, :]

        if imgs[0].shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0])
        if imgs[0].shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, imgs[0].shape[1] - self.crop_size[1])

        for i in range(len(imgs)):
            imgs[i] = imgs[i][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        for i in range(len(flows)):
            flows[i] = flows[i][
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        for i in range(len(valids)):
            valids[i] = valids[i][
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        return imgs, flows, valids

    def __call__(self, imgs, flows, valids):
        imgs = self.color_transform(imgs)
        imgs = self.eraser_transform(imgs)

        if self.pwc_aug:
            raise NotImplementedError
        else:
            imgs, flows, valids = self.spatial_transform(imgs, flows, valids)

        imgs = [np.ascontiguousarray(imgs[i]) for i in range(len(imgs))]
        flows = [np.ascontiguousarray(flows[i]) for i in range(len(flows))]
        valids = [np.ascontiguousarray(valids[i]) for i in range(len(valids))]

        return imgs, flows, valids


class SparseFlowAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        do_flip=False,
        do_rotate=False,
        pre_scale=0.0,
    ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.pre_scale = pre_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.do_rotate = do_rotate
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.rotate_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):
        cnt_imgs = len(imgs)
        image_stack = np.concatenate(imgs, axis=0)
        image_stack = np.array(
            self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
        )
        imgs = list(np.split(image_stack, cnt_imgs, axis=0))
        return imgs

    def eraser_transform(self, imgs):
        ht, wd = imgs[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(imgs[-1].reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                imgs[-1][y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return imgs

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht), indexing="xy")
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, imgs, flows, valids):
        if self.pre_scale != 0:
            cur_scale = 2**self.pre_scale

            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            for i in range(len(flows)):
                flows[i] = cv2.resize(
                    flows[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                flows[i] = flows[i] * [cur_scale, cur_scale]

            for i in range(len(valids)):
                valids[i] = cv2.resize(
                    valids[i],
                    None,
                    fx=cur_scale,
                    fy=cur_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                valids[i] = valids[i].round()

        if self.do_rotate:
            if np.random.rand() < self.rotate_prob:
                if np.random.rand() < 0.5:
                    num_rot = 1
                else:
                    num_rot = -1

                for i in range(len(imgs)):
                    imgs[i] = np.rot90(imgs[i], k=num_rot)

                for i in range(len(flows)):
                    if num_rot == 1:
                        flows[i] = np.rot90(flows[i], k=1)[:, :, ::-1] * [-1.0, 1.0]
                    elif num_rot == -1:
                        flows[i] = np.rot90(flows[i], k=-1)[:, :, ::-1] * [1.0, -1.0]

                for i in range(len(valids)):
                    valids[i] = np.rot90(valids[i], k=num_rot)

        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > imgs[0].shape[0]:
            pad_b = self.crop_size[0] - imgs[0].shape[0]
        if self.crop_size[1] > imgs[0].shape[1]:
            pad_r = self.crop_size[1] - imgs[0].shape[1]
        if pad_b != 0 or pad_r != 0:
            for i in range(len(imgs)):
                imgs[i] = np.pad(
                    imgs[i],
                    ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                    "constant",
                    constant_values=((0, 0), (0, 0), (0, 0)),
                )

            for i in range(len(flows)):
                flows[i] = np.pad(
                    flows[i],
                    ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                    "constant",
                    constant_values=((0, 0), (0, 0), (0, 0)),
                )

            for i in range(len(valids)):
                valids[i] = np.pad(
                    valids[i],
                    ((pad_t, pad_b), (pad_l, pad_r)),
                    "constant",
                    constant_values=((0, 0), (0, 0)),
                )
        # randomly sample scale

        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images

            for i in range(len(imgs)):
                imgs[i] = cv2.resize(
                    imgs[i],
                    None,
                    fx=scale_x,
                    fy=scale_y,
                    interpolation=cv2.INTER_LINEAR,
                )

            for i in range(len(flows)):
                flow, valid = self.resize_sparse_flow_map(
                    flows[i], valids[i], fx=scale_x, fy=scale_y
                )
                flows[i] = flow
                valids[i] = valid

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                for i in range(len(imgs)):
                    imgs[i] = imgs[i][:, ::-1]

                for i in range(len(flows)):
                    flows[i] = flows[i][:, ::-1] * [-1.0, 1.0]

                for i in range(len(valids)):
                    valids[i] = valids[i][:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(
            -margin_x, imgs[0].shape[1] - self.crop_size[1] + margin_x
        )

        y0 = np.clip(y0, 0, imgs[0].shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, imgs[0].shape[1] - self.crop_size[1])

        for i in range(len(imgs)):
            imgs[i] = imgs[i][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        for i in range(len(flows)):
            flows[i] = flows[i][
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        for i in range(len(valids)):
            valids[i] = valids[i][
                y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
            ]

        return imgs, flows, valids

    def __call__(self, imgs, flows, valids):
        imgs = self.color_transform(imgs)
        imgs = self.eraser_transform(imgs)

        imgs, flows, valids = self.spatial_transform(imgs, flows, valids)

        imgs = [np.ascontiguousarray(imgs[i]) for i in range(len(imgs))]
        flows = [np.ascontiguousarray(flows[i]) for i in range(len(flows))]
        valids = [np.ascontiguousarray(valids[i]) for i in range(len(valids))]

        return imgs, flows, valids
