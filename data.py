import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tf
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    IB = InterpolationMode.BICUBIC
except ImportError:
    IB = Image.BICUBIC


def saveImage(
    x,
    save_dir,
    name=["Untitled"],
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    resize=None,
):
    assert isinstance(x, torch.Tensor)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    b_size = x.size(0)
    if isinstance(name, str):
        name = [name]
    assert isinstance(name, list) and len(name) == b_size
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for i in range(b_size):
        x_i = x[i, :, :, :]

        x_i = np.array(x_i).transpose(1, 2, 0)
        x_i = x_i * np.array(std) + np.array(mean)
        np.clip(x_i, 0, 1)
        x_i *= 255

        if resize is not None:
            assert isinstance(resize, list)
            x_i = Image.fromarray(x_i.astype(np.uint8))
            for sz in resize:
                x_i = x_i.resize((sz, sz), Image.BICUBIC)
            x_i = np.array(x_i)

        cvx = np.zeros_like(x_i)
        cvx[:, :, 0] = x_i[:, :, 2]
        cvx[:, :, 1] = x_i[:, :, 1]
        cvx[:, :, 2] = x_i[:, :, 0]
        cv2.imwrite(os.path.join(save_dir, name[i] + ".png"), cvx)


class ImageFile(Dataset):
    def __init__(self, img_dir, ds_size=None, out_size=None):
        super().__init__()

        self.img_dir = img_dir
        self.images = sorted(os.listdir(self.img_dir))

        self.ds_size = tuple(
            (ds_size, ds_size)) if ds_size is not None else None
        self.out_size = tuple(
            (out_size, out_size)) if out_size is not None else None

        self.transformer_init()

    def transformer_init(self):
        tf_list = []
        if self.out_size is not None:
            tf_list.append(tf.Resize(self.out_size, IB))
        tf_list.append(tf.ToTensor())
        tf_list.append(tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.tf = tf.Compose(tf_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]

        img = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")

        # Y = up(ds(X) + N)
        if self.ds_size is not None:
            img = img.resize((self.ds_size), Image.BICUBIC)

        res = self.tf(img)

        return res
