import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from datasets import build_dataset
from utils.dds_utils import image_optimization
from utils.ptp_utils import (
    AttentionStore,
    aggregate_cross_att,
    aggregate_self_64,
    aggregate_self_att,
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

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/voc12/main.yaml")

    args, unknown = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown)

    device = torch.device(config.device)
    same_seeds(config.seed)
    dataset = build_dataset(config)
    img_output_path = os.path.join(config.output_path, "images")
    os.makedirs(img_output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "config.yaml"))

    dtype = torch.float16 if config.dtype == "fp16" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.diffusion_path, torch_dtype=dtype
    ).to(device)

    controller = AttentionStore()
    register_attention_control(pipeline, controller, config)

    if config.use_blip:
        blip_processor = BlipProcessor.from_pretrained(config.blip_path)
        blip_model = BlipForConditionalGeneration.from_pretrained(config.blip_path)
        blip_model.to(device)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (img, label, name) in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing images..."
    ):
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        y = torch.where(label)[0]
        img_512 = np.array(img.resize((512, 512), resample=Image.BILINEAR))

        # generate mask for each label in the image·
        for i in range(y.shape[0]):
            # 1. prompts generation
            cls_name = config.category[y[i].item()]
            cls_name_len = len(pipeline.tokenizer.encode(cls_name)) - 2
            pos = [4 + i for i in range(cls_name_len)]
            text_source = f"a photograph of {cls_name}."
            text_target = f"a photograph of ''."

            if config.use_blip:
                with torch.inference_mode():
                    # blip inference
                    blip_inputs = blip_processor(
                        img, text_source[:-1], return_tensors="pt"
                    ).to(device)
                    blip_out = blip_model.generate(**blip_inputs)
                    blip_out_prompt = blip_processor.decode(
                        blip_out[0], skip_special_tokens=True
                    )
                    # update pos for cross attention extraction
                    next_word = blip_out_prompt[len(text_source) :].split(" ")[0]
                    if next_word.endswith("ing"):
                        pos.append(pos[-1] + 1)
                    # refer to clip-es (CVPR 2023), we add background categories to the prompt
                    # related code: https://github.com/linyq2117/CLIP-ES/blob/main/clip_text.py
                    # paper: https://arxiv.org/abs/2212.09506
                    text_source = (
                        text_source[:-1]
                        + "++"
                        + blip_out_prompt[len(text_source) - 1 :]
                        + " and "
                        + ",".join(config.bg_category)
                        + "."
                    )
                    text_target = (
                        text_target[:-1]
                        + blip_out_prompt[len(text_source) - 1 :]
                        + " and "
                        + ",".join(config.bg_category)
                        + "."
                    )

            if config.print_prompt:
                tqdm.write(f"image: {k}, source_text: {text_source}")
                tqdm.write(f"image: {k}, target_text: {text_target}")

            # 2. dds loss optimization
            controller.reset()
            image_optimization(pipeline, img_512, text_source, text_target, config)

            # 3. refine attention map
            att_map = aggregate_cross_att(controller, pos, config)

            self_att = aggregate_self_att(controller, config)
            for _ in range(config.self_times):
                att_map = torch.matmul(self_att, att_map)

            self_64 = aggregate_self_64(controller)
            for _ in range(config.self_64_times):
                att_map = torch.matmul(self_64, att_map)

            # 4. save attention map as mask
            mask = att_map.view(64, 64).unsqueeze(0).unsqueeze(0)
            mask: torch.Tensor = F.interpolate(mask, size=(h, w), mode="bilinear")
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
            mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(f"{img_output_path}/{k}_{cls_name}.png", mask)
