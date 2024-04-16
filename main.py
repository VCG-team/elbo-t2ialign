import os
import shutil
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from datasets import COCOClsDataset, VOC12Dataset
from utils.dds_utils import image_optimization
from utils.ptp_utils import (
    AttentionStore,
    aggregate_self_att,
    generate_att_v2,
    register_attention_control,
)


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    config_path = "./configs/voc12_main.yaml"
    config = OmegaConf.load(config_path)

    img_output_path = f"{config.output_path}/images"
    code_output_path = f"{config.output_path}/codes"
    device = torch.device(config.device)
    same_seeds(config.seed)

    os.makedirs(code_output_path, exist_ok=True)
    os.makedirs(img_output_path, exist_ok=True)

    shutil.copy(os.path.abspath(__file__), code_output_path)
    shutil.copy("./utils/ptp_utils.py", code_output_path)
    shutil.copy(config_path, code_output_path)

    if config.dataset == "voc12":
        dataset = VOC12Dataset(config.data_name_list, config.data_root)
    elif config.dataset == "coco":
        dataset = COCOClsDataset(config.data_name_list, config.data_root)
    else:
        sys.exit("Dataset not supported")

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.diffusion_path, local_files_only=True
    ).to(device)
    controller = AttentionStore()
    register_attention_control(pipeline, controller)

    if config.use_blip:
        blip_processor = BlipProcessor.from_pretrained(config.blip_path)
        blip_model = BlipForConditionalGeneration.from_pretrained(config.blip_path).to(
            device
        )

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (img, label) in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing images..."
    ):

        images = []
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        y = torch.where(label)[0]
        img_512 = np.array(img.resize((512, 512), resample=Image.BILINEAR))

        # if no label, skip the image
        if y.shape[0] == 0:
            continue

        for i in range(y.shape[0]):
            text_source = f"a photograph of {config.category[y[i].item()]}."
            text_target = f"a photograph of ''."

            if config.use_blip:
                with torch.inference_mode():
                    blip_inputs = blip_processor(
                        img, text_source[:-1], return_tensors="pt"
                    ).to(device)
                    blip_out = blip_model.generate(**blip_inputs)
                    blip_out_prompt = blip_processor.decode(
                        blip_out[0], skip_special_tokens=True
                    )
                    length = len(text_source[:-1])

                    text_source = (
                        text_source[:-1]
                        + "++"
                        + blip_out_prompt[length:]
                        + " and "
                        + ",".join(config.bg_category)
                        + "."
                    )
                    text_target = text_target[:-1] + "."
            if config.print_prompt:
                tqdm.write(f"image: {k}, source_text: {text_source}")
                tqdm.write(f"image: {k}, target_text: {text_target}")

            controller.reset()
            image_optimization(pipeline, img_512, text_source, text_target, config)

            cross_att_map = generate_att_v2(
                [text_source],
                controller,
                4,
                weight=[0.3, 0.5, 0.1, 0.1],
                cross_threshold=config.cross_threshold,
            )
            self_att = aggregate_self_att(controller).view(64 * 64, 64 * 64)

            for _ in range(config.self_times):
                cross_att_map = torch.matmul(self_att, cross_att_map)

            self_64 = (
                torch.stack([att for att in controller.attention_store["up_self"][6:9]])
                .mean(0)
                .cpu()
            )
            att_map = torch.matmul(self_64, cross_att_map).view(64, 64)

            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
            att_map = F.sigmoid(8 * (att_map - 0.4))
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

            images.append((att_map).unsqueeze(0).repeat(3, 1, 1))

        images = torch.stack(images)
        images = F.interpolate(
            images, size=(h, w), mode="bilinear", align_corners=False
        )

        for i in range(images.shape[0]):
            images[i] = (
                (images[i] - images[i].min()) / (images[i].max() - images[i].min())
            ) * 255

        for i in range(0, y.shape[0]):
            cls_name = config.category[y[i].item()]
            cv2.imwrite(
                f"{img_output_path}/{k}_{cls_name}.png",
                images[i].permute(1, 2, 0).cpu().numpy(),
            )
