# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/datasets.py
import os
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path):

    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = [cls_labels_dict[img_name] for img_name in img_name_list]

    return label_list


class SegDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
    ):
        self.name = config.dataset
        self.data_root = config.data_root
        self.img_name_list = load_img_name_list(config.data_name_list)
        # checking use cls_predict.npy or cls_labels.npy
        # see ./configs/io/io.yaml for details
        if config.get("use_cls_predict", False):
            label_file_path = os.path.join(config.output_path, "cls_predict.npy")
        else:
            label_file_path = config.name_to_cls_labels
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.img_path = config.img_path
        self.gt_path = config.gt_path
        self.category = config.category

    def __getitem__(self, idx) -> Tuple[str, str, str, torch.Tensor]:
        name = self.img_name_list[idx]
        img_path = self.img_path.format(data_root=self.data_root, img_name=name)
        gt_path = self.gt_path.format(data_root=self.data_root, img_name=name)
        label = torch.from_numpy(self.label_list[idx])
        return name, img_path, gt_path, label

    def __len__(self):
        return len(self.img_name_list)
