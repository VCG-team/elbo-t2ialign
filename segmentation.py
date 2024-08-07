import os
import shutil
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from math import sqrt

import torch
import torch.nn.functional as F
from cv2 import imwrite
from diffusers import DiffusionPipeline
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import build_dataset
from utils.cfg_utils import merge_cli_cfg
from utils.dds_utils import (
    get_image_embeddings,
    get_text_embeddings,
    image_optimization,
)
from utils.img2text import Img2Text
from utils.ptp_utils import (
    AttentionStore,
    aggregate_cross_att,
    aggregate_self_att,
    register_attention_control,
)

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

    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg)
    config = merge_cli_cfg(config, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "segmentation.yaml"))

    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_output_path = os.path.join(config.output_path, "images")
    if os.path.exists(img_output_path):
        shutil.rmtree(img_output_path)
    os.makedirs(img_output_path, exist_ok=True)
    dataset = build_dataset(config)
    category = list(config.category.keys())
    if config.use_img2text:
        img2text = Img2Text(config)

    diffusion_device = torch.device(config.diffusion.device)
    diffusion_dtype = (
        torch.float16 if config.diffusion.dtype == "fp16" else torch.float32
    )
    pipe = DiffusionPipeline.from_pretrained(
        config.diffusion.variant,
        torch_dtype=diffusion_dtype,
        use_safetensors=True,
        cache_dir=config.model_dir,
    ).to(diffusion_device)
    pipe.image_processor.config.resample = "bilinear"
    # The VAE is in float32 to avoid NaN losses
    # see: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py#L712
    pipe.vae = pipe.vae.to(torch.float32)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    controller = AttentionStore()
    register_attention_control(pipe, controller, config)

    text_emb_negative = get_text_embeddings(pipe, "")
    # if the model is SDXL architecture, get add_time_ids
    add_time_ids = None
    if text_emb_negative[1] is not None:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        img_size = pipe.vae.config.sample_size
        add_time_ids = pipe._get_add_time_ids(
            (img_size, img_size),
            (0, 0),
            (img_size, img_size),
            dtype=text_emb_negative[1].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(diffusion_device)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (img, label, name) in enumerate(tqdm(dataset, desc="processing images...")):
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        labels = torch.where(label)[0].tolist()
        z_source = get_image_embeddings(pipe, img)

        # generate mask for each label in the image
        for cls_idx in labels:
            # 1. generate text prompts
            cls_name = category[cls_idx]
            # -2 to remove [START] and [END] tokens
            cls_name_len = len(pipe.tokenizer.encode(cls_name)) - 2
            text_source = f"a photograph of {cls_name}"
            text_target = f"a photograph of {config.special_token}"
            # pos for cls_name in text_source, 1 for [START], 3 for a photograph of, 4 = 1 + 3
            pos = [4 + i for i in range(cls_name_len)]

            if config.use_img2text:
                # use large multimodal model to convert img to text
                out_text = img2text(img, name, text_source)
                # update pos for cross attention extraction
                next_word = out_text.split(" ")[0]
                if next_word.endswith("ing"):
                    pos.append(pos[-1] + 1)
                # refer to clip-es (CVPR 2023), we add background categories to the prompt
                # related code: https://github.com/linyq2117/CLIP-ES/blob/main/clip_text.py
                # paper: https://arxiv.org/abs/2212.09506
                text_target = (
                    text_target
                    + " "
                    + out_text
                    + " and "
                    + ",".join(config.bg_category)
                )
                text_source = (
                    text_source
                    + "++ "
                    + out_text
                    + " and "
                    + ",".join(config.bg_category)
                )

            # 2. prepare image optimization input
            text_emb_source = get_text_embeddings(pipe, text_source)
            text_emb_target = get_text_embeddings(pipe, text_target)
            z_target = z_source.clone()
            mask_fn = None
            if config.enable_mask:
                # if apply mask to loss, we need to def the mask function
                def mask_fn():
                    att_maps = controller.get_average_attention()
                    mask = aggregate_cross_att(att_maps, pos, config)
                    max_res = round(sqrt(mask.shape[0]))
                    return mask.view(max_res, max_res)

            # 3. dds loss optimization and attention maps collection
            controller.reset()
            z_target, time_to_eps = image_optimization(
                pipe,
                z_source,
                z_target,
                text_emb_source,
                text_emb_target,
                text_emb_negative,
                add_time_ids,
                config.loss_type,
                config.optimize_timesteps,
                defaultdict(lambda: None),
                config,
                mask_fn,
            )
            if config.delay_collection:
                if not config.collect_with_original_eps:
                    time_to_eps = defaultdict(lambda: None)
                controller.reset()
                image_optimization(
                    pipe,
                    z_source,
                    z_target,
                    text_emb_source,
                    text_emb_target,
                    text_emb_negative,
                    add_time_ids,
                    "none",
                    config.collect_timesteps,
                    time_to_eps,
                    config,
                )

            # 4. refine attention map
            att_maps = controller.get_average_attention()
            mask = aggregate_cross_att(att_maps, pos, config)
            self_att = aggregate_self_att(att_maps, config)
            mask = torch.matmul(self_att, mask)

            # 5. save attention map as mask
            max_res = round(sqrt(mask.shape[0]))
            mask = mask.view(1, 1, max_res, max_res)
            mask: torch.Tensor = F.interpolate(mask, size=(h, w), mode="bilinear")
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
            mask = mask.squeeze().cpu().numpy()
            imwrite(f"{img_output_path}/{k}_{cls_name}.png", mask)
