# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/evaluation.py
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

category = []
num_cls = 0
config = None


def load_predict_and_gt(
    predict_folder: str, data_root: str, gt_list: str
) -> List[Tuple]:
    # store image level statistics
    image_statistics = []

    # convert img idx to the predicted cls names
    idx_to_cls = defaultdict(list)
    for predict_file in os.listdir(predict_folder):
        idx = int(predict_file.split("_")[0])
        cls = predict_file.split("_")[1].split(".")[0]
        idx_to_cls[idx].append(cls)

    # ground truth filename list
    df = open(gt_list, "r")
    name_list = [name.strip() for name in df.readlines()]
    df.close()

    # load each image gt and predict in name_list order
    for idx in tqdm(range(len(name_list)), desc="get image statistics..."):
        # 1. load gt
        name = name_list[idx]
        if config.dataset == "coco":
            gt_path = os.path.join(
                data_root, "annotations", "val2017", f"{name}_labelTrainIds.png"
            )
        elif config.dataset == "voc":
            gt_path = os.path.join(data_root, "SegmentationClassAug", f"{name}.png")
        elif config.dataset == "context":
            gt_path = os.path.join(data_root, "SegmentationClassContext", f"{name}.png")
        else:
            sys.exit("Unknown dataset")
        gt = np.array(Image.open(gt_path))
        # follow GroupViT (CVPR 2022), we only use first 80 classes in COCO, set other classes to background
        # related code: https://github.com/NVlabs/GroupViT/blob/main/convert_dataset/convert_coco_object.py
        if config.dataset == "coco":
            gt = gt + 1
            gt[gt > 80] = 0

        # 2. load predict
        h, w = gt.shape
        predict = np.zeros((num_cls, h, w), np.float32)

        # get all cls predict in this image
        for cls in idx_to_cls[idx]:
            predict_file = os.path.join(predict_folder, f"{idx}_{cls}.png")
            predict_cls = np.array(Image.open(predict_file), dtype=np.float32) / 255

            # search for cls_idx
            cls_idx = 1
            for cls_name in config.category:
                if cls == str(cls_name) or cls in config.category[cls_name]:
                    break
                cls_idx += 1
            if cls_idx == num_cls:
                sys.exit(f"Unknown class: {cls}")

            predict[cls_idx] = predict_cls

        # merge all cls predict to get the final predict
        predict_value = np.max(predict, axis=0)
        predict = np.argmax(predict, axis=0).astype(np.uint8)

        # 3. save image level statistics
        image_statistics.append((gt, predict_value, predict, idx))

    return image_statistics


def apply_threshold(image_statistics: List[Tuple], threshold: float) -> List[Tuple]:
    # store instance level statistics, notice one image has multiple instances
    instance_statistics = []

    # process each image in image_statistics order
    for gt, predict_value, predict, img_idx in image_statistics:

        # image meta info
        h, w = gt.shape
        # we follow MCTFormer (CVPR 2022) to prepare VOCaug dataset annotations.
        # so we only use valid region to compute metrics in VOCaug dataset.
        # related code: https://github.com/xulianuwa/MCTformer/blob/main/evaluation.py#L40
        cal = True
        if config.dataset == "voc":
            cal = gt < 255

        # compute predicted background region by threshold
        background = predict_value < threshold
        predict_copy = predict.copy()
        predict_copy[background] = 0

        # get instance level statistics for each class
        for cls_idx in range(num_cls):

            p = (predict_copy == cls_idx) * cal
            t = gt == cls_idx
            sum_p = np.sum(p)
            sum_t = np.sum(t)

            if sum_t == 0 and sum_p == 0:
                continue

            sum_tp = np.sum(t * p)

            instance_statistics.append((sum_p, sum_t, sum_tp, h, w, cls_idx, img_idx))

    return instance_statistics


def apply_metrics(
    instance_statistics: List[Tuple],
    filter_fn: Callable = None,
    sort_fn: Callable = None,
    bucket_num: int = 1,
) -> Dict[str, List]:
    # evaluation metrics for each data bucket
    metrics = {
        "m_IoU": [],
        "m_TP_T": [],
        "m_TP_P": [],
        "m_FP": [],
        "m_FN": [],
    }

    # filter and sort instance_statistics
    if filter_fn is not None:
        instance_statistics = list(filter(filter_fn, instance_statistics))
        metrics.update({"count": len(instance_statistics)})
    if sort_fn is not None:
        instance_statistics.sort(key=sort_fn)
        metrics.update({"boundary": []})

    # divide instance_statistics into bucket_num parts
    bucket_size = len(instance_statistics) // bucket_num + 1

    # compute metrics for each bucket
    for b_idx in range(bucket_num):
        # The start and end of the b_idx data bucket
        start = b_idx * bucket_size
        end = min((b_idx + 1) * bucket_size, len(instance_statistics))

        # count predict area(P)、ground truth area(T)、intersection of T and P(TP)、class num(cls_cnt) for each class
        P = np.zeros(num_cls, dtype=np.int64)
        T = np.zeros(num_cls, dtype=np.int64)
        TP = np.zeros(num_cls, dtype=np.int64)
        cls_cnt = np.zeros(num_cls, dtype=np.int32)

        for sum_p, sum_t, sum_tp, _, _, cls_idx, _ in instance_statistics[start:end]:
            P[cls_idx] += sum_p
            T[cls_idx] += sum_t
            TP[cls_idx] += sum_tp
            cls_cnt[cls_idx] += 1

        # compute metrics
        valid_idx = cls_cnt > 0

        IoU = TP / (T + P - TP + 1e-10)
        TP_T = TP / (T + 1e-10)
        TP_P = TP / (P + 1e-10)
        FP = (P - TP) / (T + P - TP + 1e-10)
        FN = (T - TP) / (T + P - TP + 1e-10)

        m_IoU = np.mean(IoU[valid_idx]) * 100
        m_TP_T = np.mean(TP_T[valid_idx]) * 100
        m_TP_P = np.mean(TP_P[valid_idx]) * 100
        m_FP = np.mean(FP[valid_idx]) * 100
        m_FN = np.mean(FN[valid_idx]) * 100

        # save metrics
        metrics["m_IoU"].append(m_IoU)
        metrics["m_TP_T"].append(m_TP_T)
        metrics["m_TP_P"].append(m_TP_P)
        metrics["m_FP"].append(m_FP)
        metrics["m_FN"].append(m_FN)

        if sort_fn is not None:
            metrics["boundary"].append(sort_fn(instance_statistics[end - 1]))

    return metrics


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset-cfg", type=str, default="./configs/dataset/voc.yaml")
    parser.add_argument("--io-cfg", type=str, default="./configs/io/io.yaml")
    parser.add_argument(
        "--method-cfg", type=str, default="./configs/method/evaluation.yaml"
    )
    args, unknown = parser.parse_known_args()

    dataset_cfg = OmegaConf.load(args.dataset_cfg)
    io_cfg = OmegaConf.load(args.io_cfg)
    method_cfg = OmegaConf.load(args.method_cfg)
    cli_cfg = OmegaConf.from_dotlist(unknown)

    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "evaluation.yaml"))

    if config.start >= config.end:
        sys.exit("Start threshold should be less than end")

    predict_dir = os.path.join(config.output_path, "images")
    category = list(config.category.keys())
    category.insert(0, "background")
    num_cls = len(category)

    image_statistics = load_predict_and_gt(
        predict_dir, config.data_root, config.data_name_list
    )

    best_threshold = 0
    best_metrics = None
    best_instance_statistics = None

    threshold_bar = tqdm(
        range(config.start, config.end), initial=config.start, total=config.end
    )
    threshold_bar.set_description("applying threshold...")
    for i in threshold_bar:
        # current threshold value
        t = i / 100.0
        # compute mIoU for whole dataset with threshold t
        instance_statistics = apply_threshold(image_statistics, threshold=t)
        metrics = apply_metrics(instance_statistics, bucket_num=1)
        # if current mIoU is less than best mIoU, we find the best threshold
        if best_metrics is not None and metrics["m_IoU"][0] < best_metrics["m_IoU"][0]:
            break
        # record the best threshold and corresponding statistics
        best_threshold = t
        best_metrics = metrics
        best_instance_statistics = instance_statistics
        threshold_bar.set_description(
            f"threshold-{t:.3f}, mIoU-{best_metrics['m_IoU'][0]:.3f}"
        )

    print(
        f"threshold range: {config.start/100}-{config.end/100}, best threshold: {best_threshold:.3f}, best mIoU: {best_metrics['m_IoU'][0]:.3f}"
    )

    print("applying metrics...", end="")
    log = {"best threshold": best_threshold}

    # overall metrics with best threshold
    metrics = best_metrics
    log.update({"overall": metrics})

    # background class metrics with best threshold
    metrics = apply_metrics(best_instance_statistics, filter_fn=lambda x: x[5] == 0)
    log.update({"background": metrics})

    # evaluate the results of all foreground classes at the best threshold
    # sorted by the ratio of gt to img size
    # divide the data into blocks
    metrics = apply_metrics(
        best_instance_statistics,
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: x[1] / (x[3] * x[4]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t_area)": metrics})

    # evaluate the results of all foreground classes at the best threshold
    # sorted by the size of gt
    # divide the data into blocks
    metrics = apply_metrics(
        best_instance_statistics,
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: int(x[1]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t)": metrics})

    metrics_output_path = os.path.join(config.output_path, "segmentation_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(log, f, indent=4)

    print("done")
