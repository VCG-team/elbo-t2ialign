# Modified from MCTFormer(CVPR 2022) https://github.com/xulianuwa/MCTformer/blob/main/evaluation.py
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

category = []
num_cls = 0
config = None


def load_predict_and_gt(
    predict_folder: str, gt_folder: str, gt_list: str
) -> List[Tuple]:
    # 用于存放图像的信息
    image_statistics = []

    # 将图像ID映射到由该图像预测出的类别名称
    idx_to_cls = defaultdict(list)
    for predict_file in os.listdir(predict_folder):
        idx = int(predict_file.split("_")[0])
        cls = predict_file.split("_")[1].split(".")[0]
        idx_to_cls[idx].append(cls)

    # gt的文件名列表
    df = pd.read_csv(gt_list, names=["filename"])
    name_list = df["filename"].values

    # 按照name_list顺序处理每张图像
    for idx in tqdm(range(len(name_list)), desc="get image statistics..."):
        name = name_list[idx]
        if config.dataset == "coco":
            gt_path = os.path.join(gt_folder, f"{int(name[-12:])}.png")
        else:
            gt_path = os.path.join(gt_folder, f"{name}.png")
        gt = np.array(Image.open(gt_path))
        h, w = gt.shape
        predict = np.zeros((num_cls, h, w), np.float32)

        # 获取该图像预测出的每个类别的预测结果
        for cls in idx_to_cls[idx]:
            predict_file = os.path.join(predict_folder, f"{idx}_{cls}.png")
            predict_cls = (
                np.array(Image.open(predict_file), dtype=np.float32)[:, :, 0] / 255
            )
            cls_idx = category.index(cls)
            predict[cls_idx] = predict_cls

        # 将不同类别的预测结果合并，得到最终的预测结果
        predict_value = np.max(predict, axis=0)
        predict = np.argmax(predict, axis=0).astype(np.uint8)
        cal = gt < 255

        image_statistics.append((predict, predict_value, gt, cal, h, w, idx))

    return image_statistics


def apply_threshold(
    image_statistics: List[Tuple], threshold: float = 0.5
) -> Dict[str, Dict]:
    # 用于存放实例的信息
    instance_statistics = []

    # 按照image_statistics顺序处理每张图像
    for predict, predict_value, gt, cal, h, w, img_idx in image_statistics:

        # 根据阈值给出图像预测结果中的背景部分
        background = predict_value < threshold
        predict_copy = predict.copy()
        predict_copy[background] = 0

        # 将图像级别的预测结果重新整理成实例级别的预测结果
        for cls_idx in range(num_cls):

            p_i = (predict_copy == cls_idx) * cal
            t_i = gt == cls_idx
            sum_p_i = np.sum(p_i)
            sum_t_i = np.sum(t_i)

            if sum_t_i == 0 and sum_p_i == 0:
                continue

            sum_tp_i = np.sum(t_i * p_i)

            instance_statistics.append(
                (sum_p_i, sum_t_i, sum_tp_i, h, w, cls_idx, img_idx)
            )

    return instance_statistics


def apply_metrics(
    instance_statistics,
    filter_fn: Callable = None,
    sort_fn: Callable = None,
    bucket_num: int = 1,
):
    # 最终返回的评估指标信息
    metrics = {
        "m_IoU": [],
        "m_TP_T": [],
        "m_TP_P": [],
        "m_FP": [],
        "m_FN": [],
    }

    # 数据的筛选和排序
    if filter_fn is not None:
        instance_statistics = list(filter(filter_fn, instance_statistics))
        metrics.update({"count": len(instance_statistics)})
    if sort_fn is not None:
        instance_statistics.sort(key=sort_fn)
        metrics.update({"boundary": []})

    # 将实例级别的预测结果分成bucket_num个块
    bucket_size = len(instance_statistics) // bucket_num + 1

    for b_idx in range(bucket_num):
        # 第b_idx个数据块的起点和终点
        start = b_idx * bucket_size
        end = min((b_idx + 1) * bucket_size, len(instance_statistics))

        # 统计每个bucket下每个类的predict区域(P)、ground truth区域(T)、T和P的交集区域(TP)、实例数量(cls_cnt)
        P = np.zeros(num_cls, dtype=np.int64)
        T = np.zeros(num_cls, dtype=np.int64)
        TP = np.zeros(num_cls, dtype=np.int64)
        cls_cnt = np.zeros(num_cls, dtype=np.int32)

        for (
            sum_p_i,
            sum_t_i,
            sum_tp_i,
            height,
            width,
            cls_idx,
            img_idx,
        ) in instance_statistics[start:end]:
            P[cls_idx] += sum_p_i
            T[cls_idx] += sum_t_i
            TP[cls_idx] += sum_tp_i
            cls_cnt[cls_idx] += 1

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
    parser.add_argument("--config", type=str, default="./configs/voc12/evaluation.yaml")
    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown)

    if config.start >= config.end:
        sys.exit("Start threshold should be less than end")

    if config.dataset == "voc12":
        gt_dir = os.path.join(config.data_root, "SegmentationClassAug")
    elif config.dataset == "coco":
        gt_dir = os.path.join(config.data_root, "mask/train2014")
    else:
        sys.exit("Dataset not supported")

    predict_dir = os.path.join(config.output_path, "images")
    log_path = os.path.join(config.output_path, "eval.json")
    category = config.category
    num_cls = len(category)

    image_statistics = load_predict_and_gt(predict_dir, gt_dir, config.data_name_list)

    best_threshold = 0
    best_metrics = None
    best_instance_statistics = None

    threshold_bar = tqdm(
        range(config.start, config.end), initial=config.start, total=config.end
    )
    threshold_bar.set_description("applying threshold...")
    for i in threshold_bar:
        # 当前的阈值t
        t = i / 100.0
        # 使用阈值t计算mIoU，并记录下来
        instance_statistics = apply_threshold(image_statistics, threshold=t)
        metrics = apply_metrics(instance_statistics, bucket_num=1)
        # 如果当前mIoU小于最好的mIoU，说明已经找到最佳阈值，停止搜索
        if best_metrics is not None and metrics["m_IoU"][0] < best_metrics["m_IoU"][0]:
            break
        # 记录最优的阈值和对应的统计数据
        best_threshold = t
        best_metrics = metrics
        best_instance_statistics = instance_statistics
        threshold_bar.set_description(
            f"threshold-{t:.3f}, mIoU-{best_metrics['m_IoU'][0]:.3f}"
        )

    print(
        f"threshold range: {config.start/100}-{config.end/100}, best threshold: {best_threshold:.3f}, best mIoU: {best_metrics['m_IoU'][0]:.3f}"
    )

    print("logging...", end="")
    log = {"best threshold": best_threshold}

    # 评估最优阈值下整体数据集的结果
    metrics = best_metrics
    log.update({"overall": metrics})

    # 评估最优阈值下所有背景类的结果
    metrics = apply_metrics(best_instance_statistics, filter_fn=lambda x: x[5] == 0)
    log.update({"background": metrics})

    # 评估最优阈值下所有前景类的结果，按照gt占img的大小比例排序，并将数据分块
    metrics = apply_metrics(
        best_instance_statistics,
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: x[1] / (x[3] * x[4]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t_area)": metrics})

    # 评估最优阈值下所有前景类的结果，按照gt的大小排序，并将数据分块
    metrics = apply_metrics(
        best_instance_statistics,
        filter_fn=lambda x: x[5] != 0,
        sort_fn=lambda x: int(x[1]),
        bucket_num=config.bucket_num,
    )
    log.update({"foreground(sort by t)": metrics})

    json.dump(log, open(log_path, "w"), indent=4)
    print("done")
