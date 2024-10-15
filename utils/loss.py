# Modified from DDS(ICCV 2023) https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoPipelineForText2Image

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]


# CutLoss from CDS(CVPR 2024) https://github.com/HyelinNAM/ContrastiveDenoisingScore/blob/main/utils/loss.py#L10
class CutLoss:
    instance = None
    init_flag = False

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, n_patches=256, patch_size=1):
        if CutLoss.init_flag:
            return
        CutLoss.init_flag = True
        self.n_patches = n_patches
        self.patch_size = patch_size

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        res = int(np.sqrt(res2))

        ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)
        trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1])
            patch_ids = patch_ids[: int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(
                patch_ids, dtype=torch.long, device=ref_noise.device
            )

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            trg_sample = trg_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            loss += self.PatchNCELoss(ref_sample, trg_sample).mean()
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T=0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()

        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim)
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[
            None, :, :
        ]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device)
        )

        return loss


class DDSLoss:
    instance = None
    init_flag = False

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(
        self,
        pipe: AutoPipelineForText2Image,
        alpha_exp=0.0,
        sigma_exp=0.0,
    ):
        if DDSLoss.init_flag:
            return
        DDSLoss.init_flag = True
        self.alpha_exp = alpha_exp
        self.sigma_exp = sigma_exp
        self.dtype = pipe.dtype
        with torch.inference_mode():
            alphas = torch.sqrt(pipe.scheduler.alphas_cumprod)
            sigmas = torch.sqrt(1 - pipe.scheduler.alphas_cumprod)
        self.alphas = alphas.to(pipe.unet.device, dtype=pipe.dtype)
        self.sigmas = sigmas.to(pipe.unet.device, dtype=pipe.dtype)
        for p in pipe.unet.parameters():
            p.requires_grad = False
        self.unet = pipe.unet

    def noise_input(self, z: T, timestep: int, eps: TN = None):
        batch_size = z.shape[0]
        t = torch.full((batch_size,), timestep, device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[t, None, None, None]
        sigma_t = self.sigmas[t, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, t, alpha_t, sigma_t

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
        if pooled_source is not None:
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
        elif loss_type == "dds" or loss_type == "cds":
            grad = eps_pred_target - eps_pred_source
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        grad = grad_coef * grad
        loss = z_target * grad.clone().detach()
        if mask is None:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        else:
            loss *= mask
            loss = loss.sum() / mask.sum()
        return loss
