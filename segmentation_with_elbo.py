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
from tqdm import tqdm

from utils.attention_control import (
    AttentionStoreHook,
    AttnProcessor,
    JointAttnProcessor,
    aggregate_cross_att,
    aggregate_self_att,
)
from utils.datasets import SegDataset
from utils.diffusion import Diffusion
from utils.img2text import Img2Text
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


class AvgAttentionStoreHook(AttentionStoreHook):
    @torch.inference_mode
    def forward(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        """
        q: (b, h, i, d)
        k, v: (b, h, j, d)
        sim: (b, h, i, j)
        out: (b, i, n) | ((b, i, n), (b, j, m))
        """
        sim_mean = sim.mean(dim=1)
        # sd3 like MMDiT attention will return a tuple
        if isinstance(out, tuple):
            img_len = out[0].shape[1]
            txt_len = out[1].shape[1]
            self.step_store["cross_att"].append(sim_mean[:, :img_len, -txt_len:])
            self.step_store["self_att"].append(sim_mean[:, :img_len, :img_len])
        else:
            key = "cross_att" if attn.is_cross_attention else "self_att"
            self.step_store[key].append(sim_mean)
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
    if config.save_cross_att:
        cross_att_out_path = os.path.join(config.output_path, "cross_att")
        if os.path.exists(cross_att_out_path):
            shutil.rmtree(cross_att_out_path)
        os.makedirs(cross_att_out_path, exist_ok=True)
    if config.elbo_path:
        if not os.path.exists(config.elbo_path):
            raise FileNotFoundError(f"ELBO file not found: {config.elbo_path}")
        elbo_dict = json.load(open(config.elbo_path))
    if config.save_elbo:
        saved_elbo = {}
    dataset = SegDataset(config)
    img2text = Img2Text(config)
    category = list(config.category.keys())
    elbo_timesteps = parse_timesteps(config.elbo_timesteps)
    collect_timesteps = parse_timesteps(config.collect_timesteps)

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
        pipe.transformer.set_attn_processor(JointAttnProcessor())
    else:
        pipe.unet.set_attn_processor(AttnProcessor())
    store_hook = AvgAttentionStoreHook(pipe)
    diffusion = Diffusion(pipe)

    # Modified from DiffSegmenter(https://arxiv.org/html/2309.02773v2) inference code
    # See: https://github.com/VCG-team/DiffSegmenter/blob/main/open_vocabulary/voc12/ptp_stable_best.py#L464
    with torch.inference_mode():
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
            elbo_noise = torch.randn(
                len(elbo_timesteps), *z_source.shape[1:], device=z_source.device
            )
            if len(labels) == 0:
                continue

            # 1. get elbo of each class
            if (
                config.fix_temperature or config.elbo_strength == 1
            ):  # use fixed temperature
                elbo_min_max = torch.ones(len(labels), device=z_source.device)
                temperature = torch.pow(config.elbo_strength, elbo_min_max)
            elif config.elbo_path:  # use elbo from file
                elbo_min_max = [
                    elbo_dict[name][category[cls_idx]] for cls_idx in labels
                ]
                elbo_min_max = torch.tensor(elbo_min_max, device=z_source.device)
                temperature = torch.pow(config.elbo_strength, elbo_min_max)
            else:  # recompute elbo
                elbo = []
                for cls_idx in labels:
                    source_cls = category[cls_idx]
                    elbo_prompt = config.elbo_text.prompt.format(source_cls=source_cls)
                    elbo_gen = img2text(img, name, elbo_prompt)
                    elbo_text = config.elbo_text.template.format(
                        source_cls=source_cls, elbo_gen=elbo_gen
                    )
                    text_emb_source = diffusion.encode_prompt(elbo_text)
                    loss = 0
                    for idx, t in enumerate(elbo_timesteps):
                        loss += diffusion.get_elbo(
                            z_source, text_emb_source, t, elbo_noise[idx]
                        )
                    elbo.append(loss)
                elbo = torch.stack(elbo)
                elbo_min_max = (elbo - elbo.min()) / (elbo.max() - elbo.min() + 1e-10)
                temperature = torch.pow(config.elbo_strength, elbo_min_max)
            if config.save_elbo:
                saved_elbo[name] = {}
                for idx, cls_idx in enumerate(labels):
                    saved_elbo[name][category[cls_idx]] = elbo_min_max[idx].item()

            # 2. collect attention maps
            source_text = "a photo of " + ", ".join(
                [category[cls_idx] for cls_idx in labels]
            )
            text_emb_source = diffusion.encode_prompt(source_text)
            store_hook.reset()
            for idx, t in enumerate(collect_timesteps):
                _ = diffusion.get_elbo(
                    z_source,
                    text_emb_source,
                    t,
                    collect_noise[idx],
                    {"attention_hooks": [store_hook]},
                )

            # 3. generate mask for each class
            for idx, cls_idx in enumerate(labels):
                # get the position of cls_name occurrence in the source text
                source_cls = category[cls_idx]
                source_text_id = pipe.tokenizer.encode(source_text)
                source_cls_id = pipe.tokenizer.encode(source_cls)[1:-1]
                for start in range(len(source_text_id) - len(source_cls_id) + 1):
                    if (
                        source_text_id[start : start + len(source_cls_id)]
                        == source_cls_id
                    ):
                        pos = [start + i for i in range(len(source_cls_id))]
                        break
                # use elbo to normalize cross attention
                cross_att = aggregate_cross_att(store_hook, 0, pos, config)
                cross_att = (
                    (cross_att - cross_att.min()) / (cross_att.max() - cross_att.min())
                ) ** temperature[idx]
                if config.save_cross_att:
                    max_res = round(sqrt(cross_att.shape[0]))
                    cross_att_img = cross_att.view(1, 1, max_res, max_res)
                    cross_att_img: T = F.interpolate(
                        cross_att_img, size=(h, w), mode="bilinear"
                    )
                    cross_att_img = cross_att_img.clamp(0, 1) * 255
                    cross_att_img = cross_att_img.squeeze().cpu().numpy()
                    imwrite(f"{cross_att_out_path}/{k}_{source_cls}.png", cross_att_img)
                # use self attention to refine self attention
                self_att = aggregate_self_att(store_hook, 0, config)
                mask = torch.matmul(self_att, cross_att)
                max_res = round(sqrt(mask.shape[0]))
                mask = mask.view(1, 1, max_res, max_res)
                mask: T = F.interpolate(mask, size=(h, w), mode="bilinear")
                mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
                mask = mask.squeeze().cpu().numpy()
                imwrite(f"{mask_out_path}/{k}_{source_cls}.png", mask)

    # save elbo optionally as json
    if config.save_elbo:
        with open(f"{config.output_path}/elbo_min_max.json", "w") as f:
            json.dump(saved_elbo, f)
