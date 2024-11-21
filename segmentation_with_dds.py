import json
import os
import random
import shutil
import warnings
from math import sqrt
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import imwrite
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention
from PIL import Image
from torch.optim.sgd import SGD
from tqdm import tqdm

from utils.attention_control import (
    AttentionHook,
    AttentionStoreHook,
    AttnProcessor,
    aggregate_cross_att,
    aggregate_self_att,
)
from utils.datasets import SegDataset
from utils.diffusion import Diffusion
from utils.img2text import Img2Text
from utils.loss import CutLoss, DDSLoss
from utils.parse_args import parse_args

T = torch.Tensor
TL = List[T]


def parse_timesteps(timesteps: List[List]) -> List:
    parsed_timesteps = []
    for t in timesteps:
        if len(t) == 3:
            parsed_timesteps.extend(range(t[0], t[1], t[2]))
        else:  # len(t) == 4, random sample
            parsed_timesteps.extend([random.randint(t[1], t[2]) for _ in range(t[3])])
    return parsed_timesteps


class CutLossHook(AttentionHook):
    def forward(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        if not attn.is_cross_attention:
            attn.self_out = out[2:]
        return out


class BlendAttentionStoreHook(AttentionStoreHook):
    def __init__(self, pipe: AutoPipelineForText2Image, config):
        super().__init__(pipe)
        self.merge_type: str = config.merge_type
        self.target_factor: float = config.target_factor

    @torch.inference_mode
    def forward(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        """
        assume [source, target] forms a batch
        q: (b, h, i, d)
        k, v: (b, h, j, d)
        sim: (b, h, i, j)
        out: (b, i, n)
        """
        if attn.is_cross_attention:
            if self.merge_type == "latent":
                source_q, target_q = q.chunk(2)
                source_k = k[0:1]
                mix_q = source_q + self.target_factor * (source_q - target_q)
                sim_mix = torch.einsum("b h i d, b h j d -> b h i j", mix_q, source_k)
                mix_att = (sim_mix * attn.scale).softmax(dim=-1, dtype=sim_mix.dtype)
                self.step_store["cross_att"].append(mix_att.mean(1))
            elif self.merge_type == "attention":
                target_q = q[1:2]
                source_k = k[0:1]
                sim_target = torch.einsum(
                    "b h i d, b h j d -> b h i j", target_q, source_k
                )
                sim_target = (sim_target * attn.scale).softmax(
                    dim=-1, dtype=sim_target.dtype
                )
                sim_source = sim[0:1]
                mix_att = sim_source + self.target_factor * (sim_source - sim_target)
                self.step_store["cross_att"].append(mix_att.mean(1))
        else:
            self.step_store["self_att"].append(sim.mean(1)[0:1])
        return out


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    config = parse_args("segmentation")

    random.seed(config.seed)
    np.random.seed(config.seed)
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
    if config.save_cross_att:
        cross_att_out_path = os.path.join(config.output_path, "cross_att")
        if os.path.exists(cross_att_out_path):
            shutil.rmtree(cross_att_out_path)
        os.makedirs(cross_att_out_path, exist_ok=True)
    if config.elbo_path:
        if not os.path.exists(config.elbo_path):
            raise FileNotFoundError(f"ELBO file not found: {config.elbo_path}")
        elbo_dict = json.load(open(config.elbo_path))
    dataset = SegDataset(config)
    img2text = Img2Text(config)
    category = list(config.category.keys())
    optimize_timesteps = parse_timesteps(config.optimize_timesteps)
    collect_timesteps = parse_timesteps(config.collect_timesteps)
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
        use_safetensors=config.diffusion.use_safetensors,
        cache_dir=config.model_dir,
        device_map=config.diffusion.device_map,
    )
    # register attention processor for attention hooks
    if isinstance(pipe, StableDiffusion3Pipeline):
        pipe.transformer.set_attn_processor(AttnProcessor())
    else:
        pipe.unet.set_attn_processor(AttnProcessor())
    store_hook = BlendAttentionStoreHook(pipe, config)
    loss_hook = CutLossHook(pipe)
    diffusion = Diffusion(pipe)
    text_emb_null = diffusion.encode_prompt("")

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    for k, (name, img_path, _, label) in enumerate(
        tqdm(dataset, desc=f"segmenting images of {dataset.name}...")
    ):
        img = Image.open(img_path).convert("RGB")
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.size
        # For PIL Image, size is a tuple (width, height)
        w, h = img.size
        labels = torch.where(label)[0].tolist()
        z_source = diffusion.encode_vae_image(img)
        # use same noise for all classes at the same timestep
        collect_noise = torch.randn(
            len(collect_timesteps), *z_source.shape[1:], device=z_source.device
        )

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
            source_text_id = pipe.tokenizer.encode(source_text)
            # remove [START] and [END] tokens
            source_cls_id = pipe.tokenizer.encode(source_cls)[1:-1]
            for start in range(len(source_text_id) - len(source_cls_id) + 1):
                if source_text_id[start : start + len(source_cls_id)] == source_cls_id:
                    pos = [start + i for i in range(len(source_cls_id))]
                    break
            if pos[-1] + 1 < len(source_text_id) and pipe.tokenizer.decode(
                source_text_id[pos[-1] + 1]
            ).endswith("ing"):
                pos.append(pos[-1] + 1)

            # 3. prepare image optimization input, loss, and optimizer
            text_emb_source = diffusion.encode_prompt(source_text)
            text_emb_target = diffusion.encode_prompt(target_text)
            text_emb_source_compel = diffusion.encode_prompt(source_text_compel)
            text_emb_target_compel = diffusion.encode_prompt(target_text_compel)
            z_target = z_source.clone()
            z_target.requires_grad = True
            dds_loss = DDSLoss(config.loss_type)
            cut_loss = CutLoss(config.n_patches, config.patch_size)
            optimizer = SGD(params=[z_target], lr=config.lr)

            # 4. image optimization
            for timestep in optimize_timesteps:
                with (
                    torch.enable_grad()
                    if config.loss_type == "cds"
                    else torch.no_grad()
                ):
                    z_t_source, eps = diffusion.noise_input(z_source, timestep, None)
                    z_t_target, _ = diffusion.noise_input(z_target, timestep, eps)
                    eps_source_null, eps_target_null, eps_source, eps_target = (
                        diffusion.get_eps_prediction(
                            [z_t_source, z_t_target] * 2,
                            [timestep] * 4,
                            [
                                text_emb_null,
                                text_emb_null,
                                text_emb_source,
                                text_emb_target,
                            ],
                        ).chunk(4)
                    )
                    eps_pred_source = diffusion.classifier_free_guidance(
                        eps_source_null, eps_source, config.guidance_scale
                    )
                    eps_pred_target = diffusion.classifier_free_guidance(
                        eps_target_null, eps_target, config.guidance_scale
                    )
                optimizer.zero_grad()
                tmp_loss = dds_loss.get_loss(
                    z_target,
                    eps_pred_source,
                    eps_pred_target,
                    eps,
                )
                loss = config.dds_loss_weight * tmp_loss
                if config.loss_type == "cds":
                    tmp_loss = 0
                    for module_name, module in pipe.unet.named_modules():
                        if type(module).__name__ == "Attention":
                            if "attn1" in module_name and "up" in module_name:
                                out = module.self_out
                                tmp_loss += cut_loss.get_attn_cut_loss(
                                    out[0:1], out[1:]
                                )
                    loss += config.cut_loss_weight * tmp_loss
                loss.backward()
                optimizer.step()

            # 5. collect attention maps
            if config.ddim_inversion:
                inverted_zs = diffusion.ddim_inversion(
                    z_source, collect_timesteps, text_emb_source
                )
            store_hook.reset()
            for idx, timestep in enumerate(collect_timesteps):
                with torch.no_grad():
                    z_t_target, eps = diffusion.noise_input(
                        z_target, timestep, collect_noise[idx]
                    )
                    if config.ddim_inversion:
                        z_t_source = inverted_zs[idx]
                    else:
                        z_t_source, _ = diffusion.noise_input(z_source, timestep, eps)
                    _ = diffusion.get_eps_prediction(
                        [z_t_source, z_t_target],
                        [timestep] * 2,
                        [text_emb_source_compel, text_emb_target_compel],
                        {"attention_hooks": [store_hook]},
                    )

            # 6. refine cross attention map and optionally save cross attention as mask
            mask = aggregate_cross_att(store_hook, 0, pos, config)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            if config.elbo_path:
                mask = mask ** (config.elbo_strength ** elbo_dict[name][source_cls])
            if config.save_cross_att:
                max_res = round(sqrt(mask.shape[0]))
                cross_att = mask.view(1, 1, max_res, max_res)
                cross_att = F.interpolate(cross_att, size=(h, w), mode="bilinear")
                cross_att = cross_att.clamp(0, 1) * 255
                cross_att = cross_att.squeeze().cpu().numpy()
                imwrite(f"{cross_att_out_path}/{k}_{category[cls_idx]}.png", cross_att)
            self_att = aggregate_self_att(store_hook, 0, config)
            mask = torch.matmul(self_att, mask)

            # 7. save mask and optionally save target img
            max_res = round(sqrt(mask.shape[0]))
            mask = mask.view(1, 1, max_res, max_res)
            mask = F.interpolate(mask, size=(h, w), mode="bilinear")
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
            mask = mask.squeeze().cpu().numpy()
            imwrite(f"{mask_out_path}/{k}_{category[cls_idx]}.png", mask)
            if config.save_img:
                img = diffusion.decode_latent(z_target)[0]
                img = img.resize((w, h))
                img.save(f"{img_out_path}/{k}_{category[cls_idx]}.png")
