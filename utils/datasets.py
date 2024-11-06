# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/datasets.py
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
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


def convert_dataset(img_paths, gt_paths, idx2cls, dataset_name):
    """
    Generate dataset metainfo ({dataset_name}.yaml), dataset class labels (cls_labels.npy), dataset images' names (id.txt). See ./config/dataset for metainfo examples, ./data/{dataset_name} for class labels and images' names examples.

    Generated {dataset_name}.yaml still needs to be modified manually, such as data_root, data_name_list, img_path, gt_path, etc.

    Args:
        img_paths (list of str): List of image file paths.
        gt_paths (list of str): List of ground truth file paths, each pixel is a class index.
        idx2cls (dict): Dictionary mapping index to class names without background class.
        dataset_name (str): Name of the dataset.
    """
    metainfo, cls_labels, img_names = {}, {}, []
    sorted_idx2cls = sorted(idx2cls.items(), key=lambda x: x[0])
    cls2idx = {cls: idx for idx, cls in sorted_idx2cls}
    idx2cls = {idx: cls for idx, cls in sorted_idx2cls}
    cls2label_idx = {cls: idx for idx, cls in enumerate(idx2cls.values())}
    cls_cnt = len(sorted_idx2cls)
    data_len = len(img_paths)

    # dataset metainfo
    metainfo["dataset"] = dataset_name
    metainfo["data_root"] = ""
    metainfo["data_name_list"] = ""
    metainfo["name_to_cls_labels"] = ""
    metainfo["img_path"] = ""
    metainfo["gt_path"] = ""
    metainfo["num_cls"] = cls_cnt
    metainfo["category"] = {}
    for idx, cls in idx2cls.items():
        metainfo["category"][cls] = [cls]
    metainfo["bg_category"] = []
    metainfo["cls2idx"] = cls2idx

    # dataset class labels and images' names
    for i in range(data_len):
        assert os.path.exists(img_paths[i]) and os.path.exists(gt_paths[i])
        img_path = Path(img_paths[i])
        gt_path = Path(gt_paths[i])
        img_names.append(img_path.stem)
        labels = np.zeros(cls_cnt)
        gt = np.array(Image.open(gt_path).convert("L"))
        cls_idxes = np.unique(gt)
        for cls_idx in cls_idxes:
            if cls_idx in idx2cls:
                labels[cls2label_idx[idx2cls[cls_idx]]] = 1
        cls_labels[img_path.stem] = labels

    # export files to default folder
    os.makedirs("./configs/dataset", exist_ok=True)
    metainfo_path = f"./configs/dataset/{dataset_name}.yaml"
    metainfo_config = OmegaConf.create(metainfo)
    OmegaConf.save(metainfo_config, metainfo_path)

    os.makedirs(f"./data/{dataset_name}", exist_ok=True)
    cls_labels_path = f"./data/{dataset_name}/cls_labels.npy"
    np.save(cls_labels_path, cls_labels, allow_pickle=True)

    img_names_path = f"./data/{dataset_name}/id.txt"
    with open(img_names_path, "w") as f:
        for img_name in img_names:
            f.write(f"{img_name}\n")


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
