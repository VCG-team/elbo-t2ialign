import os
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
    DiffusionPipeline,
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
from utils.datasets import build_dataset
from utils.img2text import Img2Text
from utils.loss import CutLoss, DDSLoss

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]


@torch.inference_mode()
def get_text_embeddings(pipe: DiffusionPipeline, text: str) -> Tuple[T, TN]:
    # use compel to do prompt weighting
    # related docs: https://huggingface.co/docs/diffusers/v0.27.2/en/using-diffusers/weighted_prompts
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
def get_image_embeddings(pipe: DiffusionPipeline, image: Image):
    vae = pipe.vae
    size, scaling_factor = vae.config.sample_size, vae.config.scaling_factor
    img_tensor = pipe.image_processor.preprocess(image, size, size)
    img_tensor = img_tensor.to(vae.device, dtype=vae.dtype)
    z_tensor = vae.encode(img_tensor)["latent_dist"].mean * scaling_factor
    return z_tensor.to(pipe.device, dtype=pipe.dtype)


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
        ).to(diffusion_device)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (img, label, name) in enumerate(tqdm(dataset, desc="segmenting images...")):
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        labels = torch.where(label)[0].tolist()
        z_source = get_image_embeddings(pipe, img)

        # generate mask for each label in the image
        for cls_idx in labels:
            # 1. generate text prompts
            cls_name = category[cls_idx]
            # miuns 2 to remove [START] and [END] tokens
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

            # 2. prepare image optimization input, loss, and optimizer
            text_emb_source = get_text_embeddings(pipe, text_source)
            text_emb_target = get_text_embeddings(pipe, text_target)
            z_target = z_source.clone()
            z_target.requires_grad = True
            dds_loss = DDSLoss(pipe, config.alpha_exp, config.sigma_exp)
            cut_loss = CutLoss(config.n_patches, config.patch_size)
            optimizer = SGD(params=[z_target], lr=config.lr)

            # 3. image optimization and attention maps collection
            time_to_eps = defaultdict(lambda: None)
            controller.reset()
            for timestep in config.optimize_timesteps:
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
                        max_res = round(sqrt(mask.shape[0]))
                        mask = mask.view(1, 1, max_res, max_res)
                        mask = F.interpolate(
                            mask, size=z_target.shape[2:], mode="bilinear"
                        )
                        mask = (mask - mask.min()) / (mask.max() - mask.min())
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
                for timestep in config.collect_timesteps:
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
