# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/datasets.py
import os

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path):

    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []

    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + ".jpg"
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])

    return label_list


class VOC12Dataset(Dataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        label_file_path="./data/voc12/voc12_cls_labels.npy",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.voc12_root = voc12_root
        self.train = "train" in img_name_list_path

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(
            os.path.join(self.voc12_root, "JPEGImages", name + ".jpg")
        ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        return img, label, name

    def __len__(self):
        return len(self.img_name_list)

class VOC10Dataset(Dataset):
    def __init__(
        self, 
        img_name_list_path, 
        voc10_root,
        label_file_path="./data/voc10/voc10_cls_labels.npy",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
            )
        self.voc10_root = voc10_root
        self.train = "train" in img_name_list_path

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(
            os.path.join(self.voc10_root, "JPEGImages", name + ".jpg")
        ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        return img, label, name

    def __len__(self):
        return len(self.img_name_list)


class COCOClsDataset(Dataset):
    def __init__(
        self,
        img_name_list_path,
        coco_root,
        label_file_path="./data/coco/coco_cls_labels.npy",
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
            img = PIL.Image.open(
                os.path.join(self.coco_root, "images/train2014", name + ".jpg")
            ).convert("RGB")
        else:
            img = PIL.Image.open(
                os.path.join(self.coco_root, "images/val2014", name + ".jpg")
            ).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        return img, label, name

    def __len__(self):
        return len(self.img_name_list)

def build_dataset(args):
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12seg':
        dataset = VOC12Dataset(img_name_list_path=args.img_list,
                                voc12_root=args.data_path, label_file_path=args.label_file_path)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list,
                                 coco_root=args.data_path, label_file_path=args.label_file_path)
        nb_classes = 80
    elif args.data_set == 'VOC10seg':
        dataset = VOC10Dataset(img_name_list_path=args.img_list,
                                voc10_root=args.data_path,label_file_path=args.label_file_path)
        nb_classes = 59

    return dataset, nb_classes
