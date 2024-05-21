import os
import shutil
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from datasets import build_dataset
from utils.dds_utils import (
    get_image_embeddings,
    get_text_embeddings,
    image_optimization,
)
from utils.ptp_utils import (
    AttentionStore,
    aggregate_cross_att,
    aggregate_self_64,
    aggregate_self_att,
    register_attention_control,
)


# fix random seed, modified from MCTFormer (CVPR 2022)
# related code: https://github.com/xulianuwa/MCTformer/blob/main/main.py#L152
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("--dataset-cfg", type=str, default="./configs/dataset/voc.yaml")
    parser.add_argument("--io-cfg", type=str, default="./configs/io/io.yaml")
    parser.add_argument(
        "--method-cfg", type=str, default="./configs/method/segmentation.yaml"
    )
    args, unknown = parser.parse_known_args()

    dataset_cfg = OmegaConf.load(args.dataset_cfg)
    io_cfg = OmegaConf.load(args.io_cfg)
    method_cfg = OmegaConf.load(args.method_cfg)
    cli_cfg = OmegaConf.from_dotlist(unknown)

    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "segmentation.yaml"))

    img_output_path = os.path.join(config.output_path, "images")
    if os.path.exists(img_output_path):
        shutil.rmtree(img_output_path)
    os.makedirs(img_output_path, exist_ok=True)

    dataset = build_dataset(config)
    category = list(config.category.keys())
    same_seeds(config.seed)

    diffusion_device = torch.device(config.diffusion.device)
    diffusion_dtype = (
        torch.float16 if config.diffusion.dtype == "fp16" else torch.float32
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.diffusion.path, torch_dtype=diffusion_dtype
    ).to(diffusion_device)
    controller = AttentionStore()
    register_attention_control(pipeline, controller, config)
    embedding_null = get_text_embeddings(pipeline, "")

    if config.use_blip:
        blip_device = torch.device(config.blip.device)
        blip_processor = BlipProcessor.from_pretrained(config.blip.path)
        blip_model = BlipForConditionalGeneration.from_pretrained(config.blip.path)
        blip_model.to(blip_device)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (img, label, name) in enumerate(tqdm(dataset, desc="processing images...")):
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        labels = torch.where(label)[0].tolist()
        z_source = get_image_embeddings(pipeline, img)

        # generate mask for each label in the image
        for cls_idx in labels:
            # 1. generate text prompts
            cls_name = category[cls_idx]
            cls_name_len = len(pipeline.tokenizer.encode(cls_name)) - 2
            pos = [4 + i for i in range(cls_name_len)]
            text_source = f"a photograph of {cls_name}."
            text_target = f"a photograph of ''."

            if config.use_blip:
                # blip inference
                with torch.inference_mode():
                    blip_inputs = blip_processor(
                        img, text_source[:-1], return_tensors="pt"
                    ).to(blip_device)
                    blip_out = blip_model.generate(**blip_inputs)
                    blip_out_text = blip_processor.decode(
                        blip_out[0], skip_special_tokens=True
                    )
                # update pos for cross attention extraction
                next_word = blip_out_text[len(text_source) :].split(" ")[0]
                if next_word.endswith("ing"):
                    pos.append(pos[-1] + 1)
                # refer to clip-es (CVPR 2023), we add background categories to the prompt
                # related code: https://github.com/linyq2117/CLIP-ES/blob/main/clip_text.py
                # paper: https://arxiv.org/abs/2212.09506
                text_target = (
                    text_target[:-1]
                    + blip_out_text[len(text_source) - 1 :]
                    + " and "
                    + ",".join(config.bg_category)
                    + "."
                )
                text_source = (
                    text_source[:-1]
                    + "++"
                    + blip_out_text[len(text_source) - 1 :]
                    + " and "
                    + ",".join(config.bg_category)
                    + "."
                )

            # 2. get image and text embeddings
            z_target = z_source.clone()
            embedding_source = get_text_embeddings(pipeline, text_source)
            embedding_source = torch.stack([embedding_null, embedding_source], dim=1)
            embedding_target = get_text_embeddings(pipeline, text_target)
            embedding_target = torch.stack([embedding_null, embedding_target], dim=1)

            # 3. dds loss optimization and attention maps collection
            if len(config.timesteps) > 0:
                config.optimize_timesteps = config.timesteps
            controller.reset()
            z_target = image_optimization(
                pipeline,
                z_source,
                z_target,
                embedding_source,
                embedding_target,
                config.optimize_timesteps,
                config.loss_type,
                config,
            )
            if config.delay_collection:
                controller.reset()
                image_optimization(
                    pipeline,
                    z_source,
                    z_target,
                    embedding_source,
                    embedding_target,
                    config.collect_timesteps,
                    "none",
                    config,
                )

            # 4. refine attention map
            att_map = aggregate_cross_att(controller, pos, config)

            self_att = aggregate_self_att(controller, config)
            for _ in range(config.self_times):
                att_map = torch.matmul(self_att, att_map)

            self_64 = aggregate_self_64(controller)
            for _ in range(config.self_64_times):
                att_map = torch.matmul(self_64, att_map)

            # 5. save attention map as mask
            mask = att_map.view(1, 1, 64, 64)
            mask: torch.Tensor = F.interpolate(mask, size=(h, w), mode="bilinear")
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
            mask = mask.squeeze().cpu().numpy()
            cv2.imwrite(f"{img_output_path}/{k}_{cls_name}.png", mask)
