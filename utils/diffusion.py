from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusion3Pipeline,
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
        # pipeline
        self.pipe = pipe
        # The VAE is always in float32 to avoid NaN losses
        # see: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py#L712
        pipe.vae = pipe.vae.to(torch.float32)
        pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
        self.vae = pipe.vae
        # image processor
        pipe.image_processor.config.resample = "bilinear"
        self.image_processor = pipe.image_processor
        # custom components for different pipelines
        if isinstance(pipe, StableDiffusionPipeline):
            self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            for p in pipe.unet.parameters():
                p.requires_grad = False
            self.unet: UNet2DConditionModel = pipe.unet
            self.compel = Compel(
                tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder
            )
            self.guidance_scale = 7.5
            with torch.inference_mode():
                self.alphas = torch.sqrt(pipe.scheduler.alphas_cumprod).to(
                    pipe.unet.device, dtype=pipe.unet.dtype
                )
                self.sigmas = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(
                    pipe.unet.device, dtype=pipe.unet.dtype
                )
        elif isinstance(pipe, StableDiffusionXLPipeline):
            self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            for p in pipe.unet.parameters():
                p.requires_grad = False
            self.unet: UNet2DConditionModel = pipe.unet
            self.compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
            self.guidance_scale = 5.0
            with torch.inference_mode():
                self.alphas = torch.sqrt(pipe.scheduler.alphas_cumprod).to(
                    pipe.unet.device, dtype=pipe.unet.dtype
                )
                self.sigmas = torch.sqrt(1 - pipe.scheduler.alphas_cumprod).to(
                    pipe.unet.device, dtype=pipe.unet.dtype
                )
            self.add_time_ids = pipe._get_add_time_ids(
                (pipe.vae.config.sample_size, pipe.vae.config.sample_size),
                (0, 0),
                (pipe.vae.config.sample_size, pipe.vae.config.sample_size),
                dtype=pipe.unet.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
            ).to(pipe.unet.device)
        elif isinstance(pipe, StableDiffusion3Pipeline):
            self.scheduler = pipe.scheduler
            for p in pipe.transformer.parameters():
                p.requires_grad = False
            self.transformer: SD3Transformer2DModel = pipe.transformer
            self.guidance_scale = 7.0
            with torch.inference_mode():
                self.sigmas = self.scheduler.sigmas.to(
                    pipe.transformer.device, dtype=pipe.transformer.dtype
                )
                self.sigmas = torch.flip(self.sigmas, [0])
                self.alphas = 1 - self.sigmas
                self.timesteps = self.scheduler.timesteps
                self.timesteps = torch.flip(self.timesteps, [0])
        else:
            raise ValueError(f"Invalid pipeline type: {type(pipe)}")

    @torch.inference_mode()
    def encode_prompt(self, text: str) -> T | Tuple[T, T]:
        if isinstance(self.pipe, StableDiffusion3Pipeline):
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=text,
                prompt_2=text,
                prompt_3=text,
                do_classifier_free_guidance=False,
            )
            return prompt_embeds, pooled_prompt_embeds
        else:
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
        if isinstance(self.pipe, StableDiffusion3Pipeline):
            return z_tensor.to(self.transformer.device, dtype=self.transformer.dtype)
        else:
            return z_tensor.to(self.unet.device, dtype=self.unet.dtype)

    @torch.inference_mode()
    def decode_latent(self, latent: T) -> Image:
        latent = latent.to(self.vae.device, dtype=self.vae.dtype)
        img_tensor = self.vae.decode(
            latent / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return self.image_processor.postprocess(img_tensor)

    def noise_input(self, z_0: T, timestep: int, eps: TN = None) -> Tuple[T, T]:
        """
        returns the noised input z_t and the noise eps
        """
        batch_size = z_0.shape[0]
        t = torch.full((batch_size,), timestep, device=z_0.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z_0)
        alpha_t = self.alphas[t, None, None, None]
        sigma_t = self.sigmas[t, None, None, None]
        z_t = alpha_t * z_0 + sigma_t * eps
        return z_t, eps

    def get_model_prediction(
        self,
        z_t: List[T],
        timestep: List[int],
        text_emb: List[T] | List[Tuple[T, T]],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T:
        batch_size = len(z_t)
        z_t = torch.cat(z_t)
        timestep = torch.tensor(timestep, dtype=torch.long)
        if isinstance(self.pipe, StableDiffusionXLPipeline):
            cond = torch.cat([emb[0] for emb in text_emb])
            pooled_cond = torch.cat([emb[1] for emb in text_emb])
            add_time_ids = torch.cat([self.add_time_ids] * batch_size)
            added_cond_kwargs = {
                "text_embeds": pooled_cond,
                "time_ids": add_time_ids,
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = self.unet(
                    z_t,
                    timestep.to(device=z_t.device),
                    encoder_hidden_states=cond,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
        elif isinstance(self.pipe, StableDiffusionPipeline):
            cond = torch.cat(text_emb)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = self.unet(
                    z_t,
                    timestep.to(device=z_t.device),
                    encoder_hidden_states=cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
        elif isinstance(self.pipe, StableDiffusion3Pipeline):
            cond = torch.cat([emb[0] for emb in text_emb])
            pooled_cond = torch.cat([emb[1] for emb in text_emb])
            t = self.timesteps[timestep]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = self.transformer(
                    hidden_states=z_t,
                    timestep=t.to(device=z_t.device),
                    encoder_hidden_states=cond,
                    pooled_projections=pooled_cond,
                    return_dict=False,
                    joint_attention_kwargs=cross_attention_kwargs,
                )[0]
        else:
            raise ValueError(f"Invalid pipeline type: {type(self.pipe)}")
        assert torch.isfinite(model_output).all()
        return model_output

    def classifier_free_guidance(
        self,
        model_output_uncond: T,
        model_output_cond: T,
        guidance_scale: Optional[float] = None,
    ) -> T:
        """
        do classifier free guidance
        paper: Classifier-Free Diffusion Guidance https://openreview.net/pdf?id=qw8AKxfYbI
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        return model_output_uncond + guidance_scale * (
            model_output_cond - model_output_uncond
        )

    def step(
        self,
        cur_z_t: T,
        cur_t: int,
        next_t: int,
        model_output: T,
    ) -> T:
        # code modified from: diffusers.FlowMatchEulerDiscreteScheduler.step
        if isinstance(self.pipe, StableDiffusion3Pipeline):
            cur_sigmas = self.sigmas[cur_t]
            next_sigmas = self.sigmas[next_t]
            next_sample = cur_z_t + (next_sigmas - cur_sigmas) * model_output
        # deterministic DDIM sampling, also can be used for DDIM inversion, return next_z_t
        # paper: Null-text Inversion(CVPR 2023) https://ieeexplore.ieee.org/document/10205188
        # code modified from: https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
        else:
            alpha_prod_t = self.scheduler.alphas_cumprod[cur_t]
            alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t]
            beta_prod_t = 1 - alpha_prod_t
            if self.scheduler.config.prediction_type == "sample":
                pred_epsilon = (
                    cur_z_t - alpha_prod_t ** (0.5) * model_output
                ) / beta_prod_t ** (0.5)
            elif self.scheduler.config.prediction_type == "v_prediction":
                pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                    beta_prod_t**0.5
                ) * cur_z_t
            else:
                pred_epsilon = model_output
            next_original_sample = (
                cur_z_t - beta_prod_t**0.5 * pred_epsilon
            ) / alpha_prod_t**0.5
            next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon
            next_sample = (
                alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
            )
        return next_sample

    @torch.inference_mode
    def inversion(self, z_0: T, timesteps: List[int], cond_emb: T) -> List[T]:
        """
        denote timesteps as [t1, t2, ..., tn], invert z_0 to z_t1, z_t2, ..., z_tn sequentially
        return: [z_t1, z_t2, ..., z_tn]
        code from: https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
        """
        z_ts = []
        cur_t, cur_z = 0, z_0.clone().detach()
        for t in timesteps:
            model_output = self.get_model_prediction([cur_z], [t], [cond_emb])
            cur_z = self.step(cur_z, cur_t, t, model_output)
            cur_t = t
            z_ts.append(cur_z)
        return z_ts

    def get_elbo(
        self,
        z: T,
        text_emb: T,
        timestep: int,
        eps: TN = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Args:
            z: vae latent of an image
            text_emb: text embedding used as condition
            timestep: timestep to add noise
            eps: noise
        Returns:
            elbo: evidence lower bound (diffusion loss)
        """
        z_t, eps = self.noise_input(z, timestep, eps)
        model_output = self.get_model_prediction(
            [z_t], [timestep], [text_emb], cross_attention_kwargs
        )
        # flow matching loss, modified from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_sd3.py#L1631
        if isinstance(self.pipe, StableDiffusion3Pipeline):
            return F.mse_loss(eps - z, model_output, reduction="mean")
        # other loss function, copied from diffusers.DDIMScheduler.step
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                z_t - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * z_t - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * z_t
        else:
            pred_epsilon = model_output
        return F.mse_loss(eps, pred_epsilon, reduction="mean")

    def generate_image(
        self, prompt: str, negtive_prompt: str = "", sample_timesteps: int = 50
    ) -> Image:
        # 1. prepare latent
        channel = self.unet.config.in_channels
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size
        latent = torch.randn(
            1, channel, height, width, device=self.unet.device, dtype=self.unet.dtype
        )

        # 2. prepare text embedding
        pos_text_emb = self.encode_prompt(prompt)
        neg_text_emb = self.encode_prompt(negtive_prompt)

        # 3. prepare timesteps
        train_timesteps = self.scheduler.config.num_train_timesteps
        step_ratio = train_timesteps // sample_timesteps
        timesteps = list(range(train_timesteps - 1, 0, -step_ratio))

        # 4. reverse diffusion process
        for t in timesteps:
            model_output_cond, model_output_uncond = self.get_model_prediction(
                [latent, latent], [t, t], [pos_text_emb, neg_text_emb]
            ).chunk(2)
            model_output = self.classifier_free_guidance(
                model_output_uncond, model_output_cond
            )
            latent = self.step(latent, t, max(0, t - step_ratio), model_output)

        # 5. decode latent to image
        return self.decode_latent(latent)[0]
