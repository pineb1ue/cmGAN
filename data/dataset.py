import os
import random
import PIL
from PIL import Image
from glob import glob
from os.path import join
from typing import Tuple

import torch
import torchvision


def _pad_image(image: PIL.Image, bg_color=(0, 0, 0)):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        result = Image.new(image.mode, (w, w), bg_color)
        result.paste(image, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(image.mode, (h, h), bg_color)
        result.paste(image, ((h - w) // 2, 0))
        return result


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class
    """

    def __init__(self, root_dirs, bg_color=(0, 0, 0), transforms=None):
        self._name_classes = [
            fname
            for fname in sorted(os.listdir(root_dirs))
            if not fname.startswith(".")
        ]
        self._data = self._load_data(root_dirs)
        self._bg_color = bg_color
        self._transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self._data[idx]["path"])

        # Padding
        image = _pad_image(image, self._bg_color)

        if self._transforms is not None:
            image = self._transforms(image)

        label = self._data[idx]["label"]
        return image, label

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def num_classes(self):
        return len(set([d["label"] for d in self._data]))

    @staticmethod
    def _load_data(root_dirs):
        data = []
        first_label = 0

        fnames = [
            fname
            for fname in sorted(os.listdir(root_dirs))
            if not fname.startswith(".")
        ]
        for label_id, label in enumerate(fnames, start=first_label):
            for path in sorted(glob(join(root_dirs, label, "*"))):
                d = {"path": path, "label": label_id}
                data.append(d)

            first_label = data[-1]["label"] + 1

        print(
            "Data loaded. Data has {} images/labels ({} sets).".format(
                len(data), len(set([d["label"] for d in data]))
            )
        )

        return data


class TripletDataset(torch.utils.data.Dataset):
    """
    Triplet dataset class for domain adaptation
        Anchor: 3D, Positive: 2D, Negative: 2D
        Anchor: 2D, Positive: 3D, Negative: 3D
    """

    def __init__(
        self,
        root_dir_3d: str,
        root_dir_2d: str,
        bg_color: Tuple[int] = (0, 0, 0),
        transforms: torchvision.transforms = None,
    ):

        self._name_classes_3d = [
            fname
            for fname in sorted(os.listdir(root_dir_3d))
            if not fname.startswith(".")
        ]
        self._name_classes_2d = [
            fname
            for fname in sorted(os.listdir(root_dir_2d))
            if not fname.startswith(".")
        ]
        assert self._name_classes_3d == self._name_classes_2d

        self._data_3d, self._data_2d = self._load_data(root_dir_3d, root_dir_2d)
        self._bg_color = bg_color
        self._transforms = transforms

    def __getitem__(self, idx: int):

        anc_3d = self._data_3d[idx]
        pos_candidates_3d = [d for d in self._data_3d if d["label"] == anc_3d["label"]]
        neg_candidates_3d = [d for d in self._data_3d if d["label"] != anc_3d["label"]]
        pos_3d = random.choice(
            [d for d in pos_candidates_3d if d["path"] != anc_3d["path"]]
        )
        neg_3d = random.choice(neg_candidates_3d)

        anc_candidates_2d = [d for d in self._data_2d if d["label"] == anc_3d["label"]]
        anc_2d = random.choice(anc_candidates_2d)
        pos_candidates_2d = [d for d in self._data_2d if d["label"] == anc_3d["label"]]
        neg_candidates_2d = [d for d in self._data_2d if d["label"] == neg_3d["label"]]
        pos_2d = random.choice(
            [d for d in pos_candidates_2d if d["path"] != anc_2d["path"]]
        )
        neg_2d = random.choice(neg_candidates_2d)

        anc_img_3d, anc_img_2d = Image.open(anc_3d["path"]), Image.open(anc_2d["path"])
        pos_img_3d, pos_img_2d = Image.open(pos_3d["path"]), Image.open(pos_2d["path"])
        neg_img_3d, neg_img_2d = Image.open(neg_3d["path"]), Image.open(neg_2d["path"])

        # Padding & Transforms
        anc_img_3d, anc_img_2d = (
            _pad_image(anc_img_3d, self._bg_color),
            _pad_image(anc_img_2d, self._bg_color),
        )
        pos_img_3d, pos_img_2d = (
            _pad_image(pos_img_3d, self._bg_color),
            _pad_image(pos_img_2d, self._bg_color),
        )
        neg_img_3d, neg_img_2d = (
            _pad_image(neg_img_3d, self._bg_color),
            _pad_image(neg_img_2d, self._bg_color),
        )

        if self._transforms is not None:
            anc_img_3d, anc_img_2d = (
                self._transforms(anc_img_3d),
                self._transforms(anc_img_2d),
            )
            pos_img_3d, pos_img_2d = (
                self._transforms(pos_img_3d),
                self._transforms(pos_img_2d),
            )
            neg_img_3d, neg_img_2d = (
                self._transforms(neg_img_3d),
                self._transforms(neg_img_2d),
            )

        domain_3d = torch.tensor([1, 0]).float()
        domain_2d = torch.tensor([0, 1]).float()

        return (
            anc_img_3d,
            pos_img_3d,
            neg_img_3d,
            anc_img_2d,
            pos_img_2d,
            neg_img_2d,
            anc_3d["label"],
            domain_3d,
            domain_2d,
        )

    def __len__(self):
        return len(self._data_3d)

    @property
    def data(self):
        return self._data_3d

    @property
    def num_classes(self):
        return len(set([d["label"] for d in self._data_3d]))

    @staticmethod
    def _load_data(root_dir_3d: str, root_dir_2d: str):
        data_3d, data_2d = [], []
        first_label_3d, first_label_2d = 0, 0

        # 3D
        fnames_3d = [
            fname
            for fname in sorted(os.listdir(root_dir_3d))
            if not fname.startswith(".")
        ]
        for label_id, label in enumerate(fnames_3d, start=first_label_3d):
            for path in sorted(glob(join(root_dir_3d, label, "*"))):
                d = {"path": path, "label": label_id}
                data_3d.append(d)
            first_label_3d = data_3d[-1]["label"] + 1

        # 2D
        fnames_2d = [
            fname
            for fname in sorted(os.listdir(root_dir_2d))
            if not fname.startswith(".")
        ]
        for label_id, label in enumerate(fnames_2d, start=first_label_2d):
            for path in sorted(glob(join(root_dir_2d, label, "*"))):
                d = {"path": path, "label": label_id}
                data_2d.append(d)
            first_label_2d = data_2d[-1]["label"] + 1

        print(
            "Data loaded.\n3D Data has {} images/labels ({} sets)\n2D Data has {} images/labels ({} sets).".format(
                len(data_3d),
                len(set([d["label"] for d in data_3d])),
                len(data_2d),
                len(set([d["label"] for d in data_2d])),
            )
        )

        return data_3d, data_2d
