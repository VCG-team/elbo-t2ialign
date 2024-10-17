from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from PIL import Image

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]


class Diffusion:
    def __init__(self, pipe: AutoPipelineForText2Image):
        for p in pipe.unet.parameters():
            p.requires_grad = False
        pipe.image_processor.config.resample = "bilinear"
        # The VAE is always in float32 to avoid NaN losses
        # see: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py#L712
        pipe.vae = pipe.vae.to(torch.float32)
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
        with torch.inference_mode():
            alphas = torch.sqrt(pipe.scheduler.alphas_cumprod)
            sigmas = torch.sqrt(1 - pipe.scheduler.alphas_cumprod)
        self.pipe = pipe
        self.vae = pipe.vae
        self.unet: UNet2DConditionModel | SD3Transformer2DModel = pipe.unet
        self.alphas = alphas.to(pipe.unet.device, dtype=pipe.unet.dtype)
        self.sigmas = sigmas.to(pipe.unet.device, dtype=pipe.unet.dtype)
        if isinstance(pipe, StableDiffusionPipeline):
            self.compel = Compel(
                tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder
            )
            self.guidance_scale = 7.5
        elif isinstance(pipe, StableDiffusionXLPipeline):
            self.compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
            self.guidance_scale = 5.0
            self.add_time_ids = pipe._get_add_time_ids(
                (self.vae.config.sample_size, self.vae.config.sample_size),
                (0, 0),
                (self.vae.config.sample_size, self.vae.config.sample_size),
                dtype=self.unet.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
            ).to(self.unet.device)
        else:
            raise ValueError(f"Invalid pipeline type: {type(pipe)}")
        self.image_processor = pipe.image_processor

    @torch.inference_mode()
    def encode_prompt(self, text: str) -> T | Tuple[T, T]:
        # use compel to do prompt weighting, blend, conjunction, etc.
        # related docs: https://huggingface.co/docs/diffusers/v0.27.2/en/using-diffusers/weighted_prompts
        # compel usage: https://github.com/damian0815/compel/blob/main/doc/syntax.md
        return self.compel(text)

    @torch.inference_mode()
    def encode_vae_image(self, image: Image) -> T:
        img_tensor = self.image_processor.preprocess(
            image, self.vae.config.sample_size, self.vae.config.sample_size
        )
        img_tensor = img_tensor.to(self.vae.device, dtype=self.vae.dtype)
        z_tensor = (
            self.vae.encode(img_tensor)["latent_dist"].mean
            * self.vae.config.scaling_factor
        )
        return z_tensor.to(self.unet.device, dtype=self.unet.dtype)

    @torch.inference_mode()
    def decode_latent(self, latent: T) -> Image:
        latent = latent.to(self.vae.device, dtype=self.vae.dtype)
        img_tensor = self.vae.decode(
            latent / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return self.image_processor.postprocess(img_tensor)

    def noise_input(self, z: T, timestep: int, eps: TN = None) -> Tuple[T, T, T]:
        """
        returns the noisy input tensor, timestep tensor, and noise tensor
        """
        batch_size = z.shape[0]
        t = torch.full((batch_size,), timestep, device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[t, None, None, None]
        sigma_t = self.sigmas[t, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, t, eps

    def get_eps_prediction(
        self,
        z_t: List[T],
        timestep: List[T],
        text_emb: List[T] | List[Tuple[T, T]],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T:
        batch_size = len(z_t)
        z_t = torch.cat(z_t)
        timestep = torch.cat(timestep)
        added_cond_kwargs = None
        if isinstance(self.pipe, StableDiffusionXLPipeline):
            cond = torch.cat([emb[0] for emb in text_emb])
            pooled_cond = torch.cat([emb[1] for emb in text_emb])
            add_time_ids = torch.cat([self.add_time_ids] * batch_size)
            added_cond_kwargs = {
                "text_embeds": pooled_cond,
                "time_ids": add_time_ids,
            }
        else:
            cond = torch.cat(text_emb)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                z_t,
                timestep,
                encoder_hidden_states=cond,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
        assert torch.isfinite(e_t).all()
        return e_t

    def classifier_free_guidance(
        self, eps_uncond: T, eps_text: T, guidance_scale: Optional[float] = None
    ) -> T:
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        return eps_uncond + guidance_scale * (eps_text - eps_uncond)

    def step(
        self,
        cur_z_t: T,
        cur_t: int,
        next_t: int,
        eps_pred: T,
    ) -> T:
        pass
