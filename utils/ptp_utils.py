# Modified from Prompt-to-Prompt(ICLR 2023) https://github.com/google/prompt-to-prompt
import abc
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from einops import rearrange
from omegaconf import DictConfig

T = torch.Tensor
TL = List[T]


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(
    pipeline: StableDiffusionPipeline,
    controller: AttentionStore,
    config: DictConfig,
):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # keep variable name with original code(prompt-to-prompt)
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask

            batch_size = len(x)
            h = self.heads
            is_cross = context is not None
            context = context if is_cross else x
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
            )
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            # for stable diffusion v1.5 series, attention heads = 8
            # and in dds loss, attn uses batched input(batch size = 4)
            # so x[0] is null text and source img, x[1] is null text and target img
            # x[2] is source text and source img, x[3] is target text and target img
            if is_cross:
                source_x = x[2] + config.target_factor * (x[3] - x[2])
                source_q = self.to_q(source_x.unsqueeze(0))
                source_q = rearrange(source_q, "b n (h d) -> (b h) n d", h=h)
                source_k = k[16:24]
                sim = torch.einsum("b i d, b j d -> b i j", source_q, source_k)
                source_att = (sim * self.scale).softmax(dim=-1)
                # save cross attention between source text and source img(take mean for 8 attention heads)
                controller(source_att.mean(0), is_cross, place_in_unet)
            else:
                source_att = attn[16:24]
                # save self attention of source img(take mean for 8 attention heads)
                controller(source_att.mean(0), is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            return to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = pipeline.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def aggregate_cross_att(
    controller: AttentionStore,
    pos: List[int],
    config: DictConfig,
):
    # 1. get all att maps
    att_maps = controller.get_average_attention()

    # 2. extract cross att maps
    cross_layer_cnt = -1
    cross_atts: TL = []
    for location in ["down", "mid", "up"]:
        for item in att_maps[f"{location}_cross"]:  # item shape: (res*res, prompt len)
            cross_layer_cnt += 1
            if cross_layer_cnt not in config.valid_cross_layer:
                continue
            cross_atts.append(item)

    # 3. get cross att maps for specific words
    cross_atts = [att[:, pos].mean(1) for att in cross_atts]

    # 4. interpolate these cross att maps to 64x64, and then apply weight
    cross_atts_64 = []
    cross_weight_sum = sum(config.cross_weight)
    for idx, att in enumerate(cross_atts):
        res = round(sqrt(att.shape[0]))
        att = att.reshape(res, res).unsqueeze(0).unsqueeze(0)
        att = F.interpolate(att, size=(64, 64), mode="bilinear")
        cross_atts_64.append(att * config.cross_weight[idx] / cross_weight_sum)

    # 5. sum up these cross att maps
    return torch.stack(cross_atts_64).sum(0).view(64 * 64, 1)


def aggregate_self_att(
    controller: AttentionStore,
    config: DictConfig,
):
    # 1. get all att maps
    att_maps = controller.get_average_attention()

    # 2. extract self att maps
    self_layer_cnt = -1
    self_atts: TL = []
    for location in ["down", "mid", "up"]:
        for item in att_maps[f"{location}_self"]:  # item shape: (res*res, res*res)
            self_layer_cnt += 1
            if self_layer_cnt not in config.valid_self_layer:
                continue
            self_atts.append(item)

    # 3. interpolate these self att maps to 64x64, and then apply weight
    self_atts_64 = []
    self_weight_sum = sum(config.self_weight)
    for idx, att in enumerate(self_atts):
        res = round(sqrt(att.shape[0]))
        # refer to diffseg (CVPR 2024), we interpolate and then repeat (repeat is important)
        # code: https://github.com/google/diffseg/blob/main/diffseg/segmentor.py#L40
        # paper: https://arxiv.org/abs/2308.12469 (Attention Aggregation Section)
        att = att.reshape(res, res, res, res)
        att: T = F.interpolate(att, size=(64, 64), mode="bilinear")
        att = att.repeat_interleave(round(64 / res), dim=0)
        att = att.repeat_interleave(round(64 / res), dim=1)
        self_atts_64.append(att * config.self_weight[idx] / self_weight_sum)

    # 4. sum up these self att maps
    return torch.stack(self_atts_64).sum(0).view(64 * 64, 64 * 64)


def aggregate_self_64(controller: AttentionStore):
    # 1. get all att maps
    att_maps = controller.get_average_attention()

    # 2. take mean for self att maps with 64x64 resolution
    return torch.stack(att_maps["up_self"][6:9]).mean(0)
