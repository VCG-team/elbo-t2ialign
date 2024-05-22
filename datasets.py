# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/datasets.py
import os
import sys
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path):

    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = [cls_labels_dict[img_name] for img_name in img_name_list]

    return label_list


class VOC12Dataset(Dataset):
    def __init__(
        self,
        img_name_list_path="./data/voc/val_id.txt",
        voc12_root="./datasets/VOCdevkit/VOC2012",
        label_file_path="./data/voc/cls_labels.npy",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.voc12_root = voc12_root
        self.train = "train" in img_name_list_path

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = Image.open(
            os.path.join(self.voc12_root, "JPEGImages", f"{name}.jpg")
        ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        return img, label, name

    def __len__(self):
        return len(self.img_name_list)


class VOCContextDataset(Dataset):
    def __init__(
        self,
        img_name_list_path="./data/context/val_id.txt",
        voc10_root="./datasets/VOCdevkit/VOC2010",
        label_file_path="./data/context/cls_labels.npy",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.voc10_root = voc10_root
        self.train = "train" in img_name_list_path

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = Image.open(
            os.path.join(self.voc10_root, "JPEGImages", f"{name}.jpg")
        ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        return img, label, name

    def __len__(self):
        return len(self.img_name_list)


class COCOClsDataset(Dataset):
    def __init__(
        self,
        img_name_list_path="./data/coco/val_id.txt",
        coco_root="./datasets/coco_stuff164k",
        label_file_path="./data/coco/cls_labels.npy",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.coco_root = coco_root
        self.train = "train" in img_name_list_path

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train:
            img = Image.open(
                os.path.join(self.coco_root, "images", "train2017", f"{name}.jpg")
            ).convert("RGB")
        else:
            img = Image.open(
                os.path.join(self.coco_root, "images", "val2017", f"{name}.jpg")
            ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        return img, label, name

    def __len__(self):
        return len(self.img_name_list)


def build_dataset(config: DictConfig) -> Dataset:
    # checking use cls_predict.npy or cls_labels.npy
    # see ./configs/io/io.yaml for details
    if not config.use_cls_predict:
        label_file_path = config.name_to_cls_labels
    elif config.cls_predict == "":
        label_file_path = os.path.join(config.output_path, "cls_predict.npy")
    else:
        label_file_path = config.cls_predict
    if not os.path.exists(label_file_path):
        sys.exit("label file not found")

    # construct dataset
    name_to_cls: Dict[str, Dataset] = {
        "voc": VOC12Dataset,
        "coco": COCOClsDataset,
        "context": VOCContextDataset,
    }
    if config.dataset not in name_to_cls:
        sys.exit("dataset not supported")
    dataset_cls = name_to_cls[config.dataset]
    return dataset_cls(config.data_name_list, config.data_root, label_file_path)
