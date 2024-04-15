from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from PIL import Image


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def show_img(images, embs, y):
    fig, axs = plt.subplots(1, len(images), figsize=(20, 10))
    for i, image in enumerate(images):
        image = (image * 255).detach().cpu().numpy()
        image = image.astype(np.uint8)
        axs[i].imshow(image, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(embs[y[i].item()])
    plt.show()


def show_pre_mask(images, embs, y):
    mask = images.clone()
    m_ = torch.max(mask, 0, keepdim=True)
    bg = (1 - m_[0]) ** 2
    mask = torch.cat((mask, bg), 0)
    mask_ = torch.max(mask, 0, keepdim=True)[1]
    length = mask.shape[0]
    show_mask(mask_, length, embs, y)


def show_mask(masks, length, embs, y):
    fig, axs = plt.subplots(1, length, figsize=(20, 10))
    for i in range(length):
        mask = ((masks == i) * 255).detach().cpu().numpy()[0]
        mask = mask.astype(np.uint8)
        axs[i].imshow(mask, cmap="gray")
        axs[i].axis("off")
        title = embs[y[i].item()] if i < y.shape[0] else "background"
        axs[i].set_title(title)
    plt.show()


def show_gt(img_path, embs, y):
    img_gt_path = img_path.replace("JPEGImages", "SegmentationClass").replace(
        ".jpg", ".png"
    )
    img_gt = np.array(Image.open(img_gt_path))
    img_gt[img_gt == 255] = 0
    img_gt[img_gt == 0] = len(np.unique(img_gt)) - 1
    for i in range(1, len(np.unique(img_gt))):
        img_gt[img_gt == np.unique(img_gt)[i]] = i - 1
    show_mask(torch.tensor(img_gt).unsqueeze(0), len(np.unique(img_gt)), embs, y)
