import os
import random
import shutil
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from math import sqrt
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from compel import Compel, ReturnedEmbeddingsType
from cv2 import imwrite
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from omegaconf import OmegaConf
from PIL import Image
from torch.optim.sgd import SGD
from tqdm import tqdm

from utils.attention_control import (
    AttentionStore,
    aggregate_cross_att,
    aggregate_self_att,
    register_attention_control,
)
from utils.check_cli_input import merge_cli_cfg
from utils.datasets import SegDataset
from utils.img2text import Img2Text
from utils.loss import CutLoss, DDSLoss

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]


@torch.inference_mode()
def encode_prompt(pipe: AutoPipelineForText2Image, text: str) -> Tuple[T, TN]:
    # use compel to do prompt weighting, blend, conjunction, etc.
    # related docs: https://huggingface.co/docs/diffusers/v0.27.2/en/using-diffusers/weighted_prompts
    # compel usage: https://github.com/damian0815/compel/blob/main/doc/syntax.md
    if isinstance(pipe, StableDiffusionPipeline):
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        return compel(text), None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        return compel(text)
    else:
        raise ValueError(f"Invalid pipeline type: {type(pipe)}")


@torch.inference_mode()
def encode_vae_image(pipe: AutoPipelineForText2Image, image: Image):
    vae = pipe.vae
    size, scaling_factor = vae.config.sample_size, vae.config.scaling_factor
    img_tensor = pipe.image_processor.preprocess(image, size, size)
    img_tensor = img_tensor.to(vae.device, dtype=vae.dtype)
    z_tensor = vae.encode(img_tensor)["latent_dist"].mean * scaling_factor
    return z_tensor.to(pipe.unet.device, dtype=pipe.unet.dtype)


@torch.inference_mode()
def decode_latent(pipe: AutoPipelineForText2Image, latent: T) -> Image:
    vae = pipe.vae
    scaling_factor = vae.config.scaling_factor
    latent = latent.to(vae.device, dtype=vae.dtype)
    img_tensor = vae.decode(latent / scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(img_tensor)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # parse arguments for configuration files
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
    # parse arguments for command line configuration
    cli_args = []
    for arg in unknown:
        if "=" in arg:
            cli_args.append(arg)
        else:
            cli_args[-1] += f" {arg}"
    cli_cfg = OmegaConf.from_dotlist(cli_args)
    # merge all configurations
    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg)
    config = merge_cli_cfg(config, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "segmentation.yaml"))

    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mask_out_path = os.path.join(config.output_path, "heatmaps")
    if os.path.exists(mask_out_path):
        shutil.rmtree(mask_out_path)
    os.makedirs(mask_out_path, exist_ok=True)
    if config.save_img:
        img_out_path = os.path.join(config.output_path, "images")
        if os.path.exists(img_out_path):
            shutil.rmtree(img_out_path)
        os.makedirs(img_out_path, exist_ok=True)
    dataset = SegDataset(config)
    img2text = Img2Text(config)
    category = list(config.category.keys())
    # refer to clip-es (CVPR 2023), we add background categories to the prompt
    # related code: https://github.com/linyq2117/CLIP-ES/blob/main/clip_text.py
    # paper: https://arxiv.org/abs/2212.09506
    bg_text = ",".join(config.bg_category)

    diffusion_dtype = (
        torch.float16 if config.diffusion.dtype == "fp16" else torch.float32
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        config.diffusion.variant,
        torch_dtype=diffusion_dtype,
        use_safetensors=True,
        cache_dir=config.model_dir,
        device_map=config.diffusion.device_map,
    )
    pipe.image_processor.config.resample = "bilinear"
    # The VAE is in float32 to avoid NaN losses
    # see: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py#L712
    pipe.vae = pipe.vae.to(torch.float32)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    controller = AttentionStore()
    register_attention_control(pipe, controller, config)

    text_emb_negative = encode_prompt(pipe, "")
    if config.guidance_scale is not None:
        guidance_scale = config.guidance_scale
    elif isinstance(pipe, StableDiffusionPipeline):
        guidance_scale = 7.5
    elif isinstance(pipe, StableDiffusionXLPipeline):
        guidance_scale = 5.0
    else:
        raise ValueError("guidance_scale is not set")
    # if the model is SDXL architecture, get add_time_ids
    add_time_ids = None
    if isinstance(pipe, StableDiffusionXLPipeline):
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
        img_size = pipe.vae.config.sample_size
        add_time_ids = pipe._get_add_time_ids(
            (img_size, img_size),
            (0, 0),
            (img_size, img_size),
            dtype=text_emb_negative[1].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(pipe.unet.device)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (name, img_path, gt_path, label) in enumerate(
        tqdm(dataset, desc="segmenting images...")
    ):
        img = Image.open(img_path).convert("RGB")
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        labels = torch.where(label)[0].tolist()
        z_source = encode_vae_image(pipe, img)

        # generate mask for each label in the image
        for cls_idx in labels:
            # 1. generate source text and target text
            source_cls = category[cls_idx]
            if config.target_cls_strategy == "special_token":
                target_cls = config.special_token
            elif config.target_cls_strategy == "synonym":
                target_cls = random.choice(config.category[source_cls])
            slot = {
                "source_cls": source_cls,
                "target_cls": target_cls,
                "bg_text": bg_text,
            }
            source_prompt = config.source_text.prompt.format(**slot)
            target_prompt = config.target_text.prompt.format(**slot)
            source_gen = img2text(img, name, source_prompt)
            target_gen = img2text(img, name, target_prompt)
            slot.update({"source_gen": source_gen, "target_gen": target_gen})
            source_text = config.source_text.template.format(**slot)
            target_text = config.target_text.template.format(**slot)
            source_text_compel = config.source_text.compel_template.format(**slot)
            target_text_compel = config.target_text.compel_template.format(**slot)

            # 2. get the position of cls_name occurrence in the source text
            souce_text_id = pipe.tokenizer.encode(source_text)
            # remove [START] and [END] tokens
            source_cls_id = pipe.tokenizer.encode(source_cls)[1:-1]
            for start in range(len(souce_text_id) - len(source_cls_id) + 1):
                if souce_text_id[start : start + len(source_cls_id)] == source_cls_id:
                    pos = [start + i for i in range(len(source_cls_id))]
                    break
            if pos[-1] + 1 < len(souce_text_id) and pipe.tokenizer.decode(
                souce_text_id[pos[-1] + 1]
            ).endswith("ing"):
                pos.append(pos[-1] + 1)

            # 3. prepare image optimization input, loss, and optimizer
            text_emb_source = encode_prompt(pipe, source_text_compel)
            text_emb_target = encode_prompt(pipe, target_text_compel)
            z_target = z_source.clone()
            z_target.requires_grad = True
            dds_loss = DDSLoss(pipe, config.alpha_exp, config.sigma_exp)
            cut_loss = CutLoss(config.n_patches, config.patch_size)
            optimizer = SGD(params=[z_target], lr=config.lr)

            # 4. image optimization and attention maps collection
            time_to_eps = defaultdict(lambda: None)
            controller.reset()
            optimize_timesteps = [
                timestep
                for start, end, step in config.optimize_timesteps
                for timestep in range(start, end, step)
            ]
            for timestep in optimize_timesteps:
                with (
                    torch.enable_grad()
                    if config.loss_type == "cds"
                    else torch.no_grad()
                ):
                    z_t_source, eps, t, alpha_t, sigma_t = dds_loss.noise_input(
                        z_source, None, timestep
                    )
                    z_t_target, _, _, _, _ = dds_loss.noise_input(
                        z_target, eps, timestep
                    )
                    time_to_eps[timestep] = eps
                    eps_pred_source, eps_pred_target = dds_loss.get_eps_prediction(
                        z_t_source,
                        z_t_target,
                        t,
                        text_emb_source,
                        text_emb_target,
                        text_emb_negative,
                        add_time_ids,
                        guidance_scale,
                    )
                    mask = None
                    if config.enable_mask:
                        att_maps = controller.get_average_attention()
                        mask = aggregate_cross_att(att_maps, pos, config)
                        self_att = aggregate_self_att(att_maps, config)
                        mask = torch.matmul(self_att, mask)
                        max_res = round(sqrt(mask.shape[0]))
                        mask = mask.view(1, 1, max_res, max_res)
                        mask = F.interpolate(
                            mask, size=z_target.shape[2:], mode="bilinear"
                        )
                        mask = (mask - mask.min()) / (mask.max() - mask.min())
                        mask[mask >= config.mask_threshold] = 1.0
                        mask[mask < config.mask_threshold] = 0.0
                optimizer.zero_grad()
                tmp_loss = dds_loss.get_loss(
                    z_target,
                    alpha_t,
                    sigma_t,
                    eps_pred_source,
                    eps_pred_target,
                    eps,
                    config.loss_type,
                    mask,
                )
                loss = config.dds_loss_weight * tmp_loss
                if config.loss_type == "cds":
                    tmp_loss = 0
                    for name, module in pipe.unet.named_modules():
                        if type(module).__name__ == "Attention":
                            if hasattr(module, "self_out"):
                                out = module.self_out
                                tmp_loss += cut_loss.get_attn_cut_loss(
                                    out[0:1], out[1:]
                                )
                    loss += config.cut_loss_weight * tmp_loss
                loss.backward()
                optimizer.step()
            if config.delay_collection:
                controller.reset()
                collect_timesteps = [
                    timestep
                    for start, end, step in config.collect_timesteps
                    for timestep in range(start, end, step)
                ]
                for timestep in collect_timesteps:
                    with torch.no_grad():
                        z_t_source, eps, t, alpha_t, sigma_t = dds_loss.noise_input(
                            z_source, time_to_eps[timestep], timestep
                        )
                        z_t_target, _, _, _, _ = dds_loss.noise_input(
                            z_target, eps, timestep
                        )
                        _, _ = dds_loss.get_eps_prediction(
                            z_t_source,
                            z_t_target,
                            t,
                            text_emb_source,
                            text_emb_target,
                            text_emb_negative,
                            add_time_ids,
                            guidance_scale,
                        )

            # 5. refine attention map as mask
            att_maps = controller.get_average_attention()
            mask = aggregate_cross_att(att_maps, pos, config)
            self_att = aggregate_self_att(att_maps, config)
            mask = torch.matmul(self_att, mask)

            # 6. save mask and target img
            max_res = round(sqrt(mask.shape[0]))
            mask = mask.view(1, 1, max_res, max_res)
            mask: T = F.interpolate(mask, size=(h, w), mode="bilinear")
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
            mask = mask.squeeze().cpu().numpy()
            imwrite(f"{mask_out_path}/{k}_{category[cls_idx]}.png", mask)
            if config.save_img:
                img = decode_latent(pipe, z_target)[0]
                img = img.resize((w, h))
                img.save(f"{img_out_path}/{k}_{category[cls_idx]}.png")
