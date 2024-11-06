# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/evaluation.py
import json
import os
import shutil
import sys
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import denseCRF
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from utils.datasets import SegDataset
from utils.parse_args import parse_args

category = []
num_cls = 0
config = None


# denseCRF copied from https://github.com/vpulab/ovam/blob/main/experiments/evaluation/ovam_sa/evaluation_ovam_sa.ipynb
# use the same param as OVAM(CVPR 2024) https://github.com/vpulab/ovam
def densecrf(I, P):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices.
    """
    w1 = 10.0  # weight of bilateral term
    alpha = 80  # spatial std
    beta = 13  # rgb  std
    w2 = 3.0  # weight of spatial term
    gamma = 3  # spatial std
    it = 5.0  # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    out = denseCRF.densecrf(I, P, param)
    return out


def load_gt_and_predict(predict_folder: str, config: DictConfig) -> List[Tuple]:
    # convert img idx to the predicted cls names
    idx_to_cls = defaultdict(list)
    for predict_file in os.listdir(predict_folder):
        # predict_file format: {idx}_{cls_name}.png
        idx = int(predict_file.split("_")[0])
        cls = predict_file.split("_")[1].split(".")[0].strip()
        idx_to_cls[idx].append(cls)
    # load dataset
    dataset = SegDataset(config)

    def process(idx):
        # 1. load ground truth
        _, img_path, gt_path, _ = dataset[idx]
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
            # search cls_idx
            cls_idx = 1
            for cls_name in config.category:
                if (cls == str(cls_name).strip()) or (
                    cls in [n.strip() for n in config.category[cls_name]]
                ):
                    break
                cls_idx += 1
            if cls_idx == num_cls:
                sys.exit(f"unknown class: {cls}")
            predict[cls_idx] = predict_cls
        # merge all cls predict to get the final predict
        predict_value = np.max(predict, axis=0)
        predict = np.argmax(predict, axis=0).astype(np.uint8)

        return (gt, predict_value, predict, idx, img_path)

    # use joblib with tqdm to load gt and predict in parallel
    # see: https://stackoverflow.com/a/77948954/12389770
    results = Parallel(n_jobs=config.n_jobs, return_as="generator")(
        delayed(process)(idx) for idx in range(len(dataset))
    )
    return list(tqdm(results, desc="loading gt and predict...", total=len(dataset)))


def apply_threshold(
    gt_and_predict: List[Tuple], threshold: float, enable_crf=False, save_mask=False
) -> List[Tuple]:

    def process(gt, predict_value, predict, img_idx, img_path):
        # store instance level statistics, notice one image has multiple instances
        instance_statistics = []

        # image meta info
        h, w = gt.shape

        # compute predicted background region by threshold
        background = predict_value < threshold
        predict_copy = predict.copy()
        predict_copy[background] = 0

        # save mask
        if save_mask:
            mask = Image.fromarray(predict_copy)
            mask.save(os.path.join(config.output_path, "masks", f"{img_idx}.png"))

        # apply denseCRF
        if enable_crf:
            img = np.array(Image.open(img_path))
            probability = np.zeros((h, w, num_cls), np.float32)
            cls_idxes = np.unique(predict_copy)
            for cls_idx in cls_idxes:
                probability[predict_copy == cls_idx, cls_idx] = 1
            predict_copy = densecrf(img, probability)

        # we follow MCTFormer (CVPR 2022) to prepare VOCaug dataset annotations.
        # so we only use valid region to compute metrics in VOCaug dataset.
        # related code: https://github.com/xulianuwa/MCTformer/blob/main/evaluation.py#L40
        if config.dataset == "voc":
            predict_copy[gt == 255] = np.array(255).astype(np.int8)

        # get instance level statistics for each class
        p = dict(zip(*np.unique(predict_copy, return_counts=True)))
        t = dict(zip(*np.unique(gt, return_counts=True)))
        tp = dict(zip(*np.unique(gt[predict_copy == gt], return_counts=True)))

        cls_idxes = (set(p.keys()) | set(t.keys())) - {255}
        for cls_idx in cls_idxes:
            sum_p = p.get(cls_idx, 0)
            sum_t = t.get(cls_idx, 0)
            sum_tp = tp.get(cls_idx, 0)
            instance_statistics.append((sum_p, sum_t, sum_tp, h, w, cls_idx, img_idx))

        return instance_statistics

    # apply threshold in parallel with joblib
    # related docs: https://joblib.readthedocs.io/en/latest/parallel.html#embarrassingly-parallel-for-loops
    results = Parallel(n_jobs=config.n_jobs, return_as="generator")(
        delayed(process)(*item) for item in gt_and_predict
    )
    # provide a progress bar for denseCRF
    if enable_crf:
        results = tqdm(
            results,
            desc="collecting metrics with denseCRF...",
            total=len(gt_and_predict),
        )
    return sum(list(results), [])


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


# use gt_and_predict and threshold to calculate the key metric.
# during evaluation, we adjust threshold to get the best key metric.
def get_key_metric(gt_and_predict: List[Tuple], threshold: float):
    instance_statistics = apply_threshold(gt_and_predict, threshold)
    metrics = apply_metrics(instance_statistics, bucket_num=1)
    key_metric = metrics["m_IoU"][0]
    return {
        "statistics": instance_statistics,
        "metrics": metrics,
        "key_metric": key_metric,
    }


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    config = parse_args("evaluation")

    if config.start >= config.end:
        sys.exit("start threshold should be less than end")
    if config.save_mask:
        mask_dir = os.path.join(config.output_path, "masks")
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)
        os.makedirs(mask_dir, exist_ok=True)
    predict_dir = os.path.join(config.output_path, "heatmaps")
    category = list(config.category.keys())
    category.insert(0, "background")
    num_cls = len(category)
    gt_and_predict = load_gt_and_predict(predict_dir, config)

    # binary search to find the best threshold for key metric
    threshold_bar = tqdm(desc="searching best threshold for key metric...")
    # 1. init left and right
    l, r = config.start, config.end
    l_info = get_key_metric(gt_and_predict, l / 100)
    r_info = l_info if l == r else get_key_metric(gt_and_predict, r / 100)
    threshold_bar.update(1)
    threshold_bar.set_description(
        f"threshold-{l/100}, key metric-{l_info['key_metric']:.3f}"
    )
    # 2. binary search
    while l < r:
        mid = (l + r) // 2
        mid_info = l_info if mid == l else get_key_metric(gt_and_predict, mid / 100)

        mid1 = mid + 1
        mid1_info = r_info if mid1 == r else get_key_metric(gt_and_predict, mid1 / 100)

        if mid_info["key_metric"] < mid1_info["key_metric"]:
            l, l_info = mid1, mid1_info
        else:
            r, r_info = mid, mid_info

        threshold_bar.update(1)
        threshold_bar.set_description(
            f"threshold-{mid/100}, key metric-{mid_info['key_metric']:.3f}"
        )
    # 3. print best threshold and key metric
    threshold_bar.close()
    print(
        f"threshold range: {config.start/100}-{config.end/100}, best threshold: {l/100}, best key metric: {l_info['key_metric']:.3f}"
    )

    # collect metrics without denseCRF
    print("collecting metrics without denseCRF...")
    log = {"best threshold": l / 100}
    # 1. overall metrics with best threshold
    metrics = apply_metrics(l_info["statistics"], bucket_num=1)
    log.update({"overall": metrics})
    # 2. classes metrics with best threshold
    for cls in category:
        metrics = apply_metrics(
            l_info["statistics"], filter_fn=lambda x: x[5] == category.index(cls)
        )
        log.update({cls: metrics})
    # 3. metrics of all foreground classes at the best threshold, sorted by the ratio of gt to img size, divide the data into blocks
    metrics = apply_metrics(
        l_info["statistics"],
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: x[1] / (x[3] * x[4]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t_area)": metrics})
    # 4. write metrics to json
    metrics_output_path = os.path.join(config.output_path, "segmentation_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(log, f, indent=4)

    # collect metrics with denseCRF
    log = {"best threshold": l / 100}
    instance_statistics = apply_threshold(
        gt_and_predict, l / 100, enable_crf=True, save_mask=config.save_mask
    )
    # 1. overall metrics with best threshold
    metrics = apply_metrics(instance_statistics, bucket_num=1)
    log.update({"overall": metrics})
    # 2. classes metrics with best threshold
    for cls in category:
        metrics = apply_metrics(
            instance_statistics, filter_fn=lambda x: x[5] == category.index(cls)
        )
        log.update({cls: metrics})
    # 3. metrics of all foreground classes at the best threshold, sorted by the ratio of gt to img size, divide the data into blocks
    metrics = apply_metrics(
        instance_statistics,
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: x[1] / (x[3] * x[4]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t_area)": metrics})
    # 4. write metrics to json
    metrics_output_path = os.path.join(
        config.output_path, "segmentation_metrics_crf.json"
    )
    with open(metrics_output_path, "w") as f:
        json.dump(log, f, indent=4)

    print("finish evaluation.")
