# Modified from Prompt-to-Prompt(ICLR 2023) https://github.com/google/prompt-to-prompt
import abc
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image
from diffusers.models.attention_processor import Attention
from einops import rearrange
from omegaconf import DictConfig

T = torch.Tensor
TL = List[T]


class AttentionHook(abc.ABC):

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        raise NotImplementedError

    def __call__(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        out = self.forward(attn, q, k, v, sim, out)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, num_att_layers: int):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = num_att_layers


# Modified from default attention processor
# link: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L716
class AttnProcessor:

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_hooks: List[AttentionHook] | None = None,
    ) -> torch.Tensor:

        h = attn.heads
        x = hidden_states
        is_cross = attn.is_cross_attention
        context = encoder_hidden_states if is_cross else x
        q = attn.to_q(x)
        k = attn.to_k(context)
        v = attn.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * attn.scale
        sim = sim.softmax(dim=-1, dtype=sim.dtype)
        out = torch.einsum("b h i j, b h j d -> b h i d", sim, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if attention_hooks is not None:
            for attention_hook in attention_hooks:
                out = attention_hook(attn, q, k, v, sim, out)
        return out


def count_diffusion_attention(pipe: AutoPipelineForText2Image):
    # return (self_att_count, cross_att_count)
    def count_att_layer(net_) -> Tuple[int, int]:
        if net_.__class__.__name__ == "Attention":
            if net_.is_cross_attention:
                return 0, 1
            return 1, 0
        s_cnt, c_cnt = 0, 0
        if hasattr(net_, "children"):
            for net__ in net_.children():
                s_cnt_, c_cnt_ = count_att_layer(net__)
                s_cnt, c_cnt = s_cnt + s_cnt_, c_cnt + c_cnt_
        return s_cnt, c_cnt

    if hasattr(pipe, "unet"):
        down_s, down_c = count_att_layer(pipe.unet.down_blocks)
        mid_s, mid_c = count_att_layer(pipe.unet.mid_block)
        up_s, up_c = count_att_layer(pipe.unet.up_blocks)
    else:
        s, _ = count_att_layer(pipe.transformer)
        n, m = s // 3, s - s // 3 * 2
        down_s, down_c, mid_s, mid_c, up_s, up_c = n, n, n, n, m, m
    return down_s, down_c, mid_s, mid_c, up_s, up_c


class AttentionStoreHook(AttentionHook):

    @staticmethod
    def get_empty_store():
        return {
            "cross_att": [],
            "self_att": [],
        }

    @torch.inference_mode
    def forward(self, attn: Attention, q: T, k: T, v: T, sim: T, out: T) -> T:
        key = "cross_att" if attn.is_cross_attention else "self_att"
        self.step_store[key].append(sim)
        return out

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.attention_store[key][i].clone()
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self) -> Dict[str, TL]:
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStoreHook, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, pipe: AutoPipelineForText2Image):
        down_s, down_c, mid_s, mid_c, up_s, up_c = count_diffusion_attention(pipe)
        super(AttentionStoreHook, self).__init__(
            down_s + down_c + mid_s + mid_c + up_s + up_c
        )
        self.step_store: Dict[str, TL] = self.get_empty_store()
        self.attention_store: Dict[str, TL] = {}
        self.cross_att_mid_layer = down_c + (mid_c - 1) / 2
        self.self_att_mid_layer = down_s + (mid_s - 1) / 2


def get_gaussian_weight(mean, sigma, length, is_cross=True):
    x = torch.arange(length, dtype=torch.float64)
    x = x - mean
    weights = torch.exp(-(x**2) / (2 * sigma**2))
    if not is_cross:
        weights = 1 - weights
    weights = weights / weights.sum() * length
    return weights


@torch.inference_mode()
def aggregate_cross_att(
    hook: AttentionStoreHook,
    batch_idx: int,
    token_pos: List[int],
    config: DictConfig,
) -> T:
    out, weight_sum, max_res = 0, 0, None
    cur_res, cur_res_idx, cur_layer = None, 0, 0
    att_maps = hook.get_average_attention()

    if config.cross_gaussian_var != 0:
        weights = get_gaussian_weight(
            hook.cross_att_mid_layer,
            config.cross_gaussian_var,
            len(att_maps["cross_att"]),
        )

    for att_map in att_maps["cross_att"]:  # attn shape: (batch, res*res, prompt len)
        res = round(sqrt(att_map.shape[1]))
        if max_res is None:
            max_res, cur_res = res, res
        if res != cur_res:
            cur_res, cur_res_idx = res, cur_res_idx + 1
        att = att_map[batch_idx, :, token_pos]
        att = att.mean(1).reshape(1, 1, res, res)
        att = F.interpolate(att, size=(max_res, max_res), mode="bilinear")
        if config.cross_gaussian_var != 0:
            weight = weights[cur_layer]
        else:
            weight = config.cross_weight[cur_res_idx]
        out += att * weight  # apply weight
        weight_sum += weight
        cur_layer += 1
    out /= weight_sum
    return out.view(max_res * max_res, 1)


@torch.inference_mode()
def aggregate_self_att(
    hook: AttentionStoreHook,
    batch_idx: int,
    config: DictConfig,
) -> T:
    out, weight_sum, max_res = (
        [0] * len(config.self_weight),
        [0] * len(config.self_weight),
        None,
    )
    cur_res, cur_res_idx, cur_layer = None, 0, 0
    att_maps = hook.get_average_attention()

    if len(config.self_gaussian_var) != 0:
        weights = [
            get_gaussian_weight(
                hook.self_att_mid_layer,
                config.self_gaussian_var[i],
                len(att_maps["self_att"]),
                False,
            )
            for i in range(len(config.self_gaussian_var))
        ]

    for att_map in att_maps["self_att"]:  # attn shape: (batch, res*res, res*res)
        res = round(sqrt(att_map.shape[1]))
        if max_res is None:
            max_res, cur_res = res, res
        if res != cur_res:
            cur_res, cur_res_idx = res, cur_res_idx + 1
        att = att_map[batch_idx]
        # refer to diffseg (CVPR 2024), we interpolate and then repeat (repeat is important)
        # related code: https://github.com/google/diffseg/blob/main/diffseg/segmentor.py#L40
        # paper: https://arxiv.org/abs/2308.12469 (Attention Aggregation Section)
        att = att.reshape(res, res, res, res)
        att = F.interpolate(att, size=(max_res, max_res), mode="bilinear")
        att = att.repeat_interleave(round(max_res / res), dim=0)
        att = att.repeat_interleave(round(max_res / res), dim=1)
        if len(config.self_gaussian_var) != 0:
            weight = [
                weights[i][cur_layer] for i in range(len(config.self_gaussian_var))
            ]
        else:
            weight = [
                config.self_weight[i][cur_res_idx]
                for i in range(len(config.self_weight))
            ]
        for i in range(len(weight)):
            out[i] += att * weight[i]  # apply weight
            weight_sum[i] += weight[i]
        cur_layer += 1
    for i in range(len(out)):
        out[i] /= weight_sum[i]
        out[i] = out[i].view(max_res * max_res, max_res * max_res)
    result = out[0]
    for i in range(1, len(out)):
        result = torch.matmul(out[i], result)
    return result
