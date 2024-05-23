# Modified from DDS(ICCV 2023) https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple, Union

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from omegaconf import DictConfig
from PIL import Image
from torch.optim.sgd import SGD

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
    size = pipe.vae.config.sample_size
    img_tensor = pipe.image_processor.preprocess(image, size, size)
    img_tensor = img_tensor.to(pipe.device, dtype=pipe.dtype)
    return pipe.vae.encode(img_tensor)["latent_dist"].mean * pipe.vae.scaling_factor


def init_pipe(device, dtype, unet, scheduler) -> Tuple[UNet2DConditionModel, T, T]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class DDSLoss:
    instance = None
    init_flag = False

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(
        self,
        pipe: DiffusionPipeline,
        alpha_exp=0,
        sigma_exp=0,
    ):
        if DDSLoss.init_flag:
            return
        DDSLoss.init_flag = True
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = alpha_exp
        self.sigma_exp = sigma_exp
        self.dtype = pipe.dtype
        self.unet, self.alphas, self.sigmas = init_pipe(
            pipe.device, pipe.dtype, pipe.unet, pipe.scheduler
        )

    @torch.inference_mode()
    def noise_input(self, z: T, eps: TN = None, timestep: Optional[int] = None):
        batch_size = z.shape[0]
        if timestep is None:
            t = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # avoid the highest timestep
                size=(batch_size,),
                device=z.device,
                dtype=torch.long,
            )
        elif isinstance(timestep, int):
            t = torch.full((batch_size,), timestep, device=z.device, dtype=torch.long)
        else:
            raise ValueError(f"Invalid timestep: {timestep}")
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[t, None, None, None]
        sigma_t = self.sigmas[t, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, t, alpha_t, sigma_t

    @torch.inference_mode()
    def get_eps_prediction(
        self,
        z_t_source: T,
        z_t_target: T,
        timestep: T,
        text_emb_source: Tuple[T, TN],
        text_emb_target: Tuple[T, TN],
        text_emb_negative: Tuple[T, TN],
        add_time_ids: TN,
        guidance_scale: float,
    ):
        # prepare inputs
        cond_source, pooled_source = text_emb_source
        cond_target, pooled_target = text_emb_target
        cond_negative, pooled_negative = text_emb_negative
        z_t = torch.cat([z_t_source, z_t_target] * 2)
        timesteps = torch.cat([timestep] * 4)
        cond = torch.cat(
            [
                cond_negative,
                cond_negative,
                cond_source,
                cond_target,
            ]
        )
        added_cond_kwargs = None
        # for SDXL, added_cond_kwargs is required
        if pooled_source:
            add_text_embeds = torch.cat(
                [
                    pooled_negative,
                    pooled_negative,
                    pooled_source,
                    pooled_target,
                ]
            )
            add_time_ids = torch.cat([add_time_ids] * 4)
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
        # forward and do classifier free guidance
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(
                z_t,
                timesteps,
                encoder_hidden_states=cond,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        # return eps_pred_source and eps_pred_target
        return e_t.chunk(2)

    def get_loss(
        self,
        z_target: T,
        alpha_t: T,
        sigma_t: T,
        eps_pred_source: T,
        eps_pred_target: T,
        eps: T,
        loss_type: str,
        mask: TN = None,
    ) -> TS:
        grad_coef = (alpha_t**self.alpha_exp) * (sigma_t**self.sigma_exp)
        if loss_type == "sds":
            grad = eps_pred_target - eps
        elif loss_type == "dds":
            grad = eps_pred_target - eps_pred_source
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        grad = grad_coef * grad
        if mask is not None:
            grad = grad * mask
        log_loss = (grad**2).mean()
        loss = z_target * grad.clone()
        loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss


def image_optimization(
    pipe: DiffusionPipeline,
    z_source: T,
    z_target: T,
    text_emb_source: Tuple[T, TN],
    text_emb_target: Tuple[T, TN],
    text_emb_negative: Tuple[T, TN],
    add_time_ids: TN,
    loss_type: str,
    timesteps: List[int],
    time_to_eps: DefaultDict[int, TN],
    config: DictConfig,
) -> Tuple[T, DefaultDict[int, TN]]:
    dds_loss = DDSLoss(pipe, config.alpha_exp, config.sigma_exp)
    time_to_eps_return = defaultdict(lambda: None)
    z_target.requires_grad = True
    optimizer = SGD(params=[z_target], lr=config.lr)
    if config.guidance_scale is not None:
        guidance_scale = config.guidance_scale
    elif isinstance(pipe, StableDiffusionPipeline):
        guidance_scale = 7.5
    else:
        guidance_scale = 5.0

    for timestep in timesteps:
        z_t_source, eps, t, alpha_t, sigma_t = dds_loss.noise_input(
            z_source, time_to_eps[timestep], timestep
        )
        z_t_target, _, _, _, _ = dds_loss.noise_input(z_target, eps, timestep)
        time_to_eps_return[timestep] = eps
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
        if loss_type == "none":
            continue
        loss, _ = dds_loss.get_loss(
            z_target,
            alpha_t,
            sigma_t,
            eps_pred_source,
            eps_pred_target,
            eps,
            loss_type,
        )
        optimizer.zero_grad()
        (config.loss_factor * loss).backward()
        optimizer.step()

    return z_target, time_to_eps_return
