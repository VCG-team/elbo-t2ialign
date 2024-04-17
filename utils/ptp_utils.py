# Modified from Prompt-to-Prompt(ICLR 2023) https://github.com/google/prompt-to-prompt
import abc
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from einops import rearrange
from omegaconf import DictConfig


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
    prompt: str,
    pos: List[int],
    config: DictConfig,
):
    # 1. get all att maps
    att_maps = controller.get_average_attention()

    # 2. extract cross att maps
    cross_layer_cnt = -1
    cross_atts = []
    for location in ["down", "mid", "up"]:
        for item in att_maps[f"{location}_cross"]:  # item shape: (res*res, prompt len)
            cross_layer_cnt += 1
            if cross_layer_cnt not in config.valid_cross_layer:
                continue
            res = round(sqrt(item.shape[0]))
            cross_maps = item.reshape(res, res, item.shape[-1])
            cross_atts.append(cross_maps)

    # 3. get cross att maps for specific words
    next_word = prompt.split(" ")[4]
    if next_word.endswith("ing"):
        cross_atts = [att[:, :, pos + [pos[-1] + 1]] for att in cross_atts]
    else:
        cross_atts = [att[:, :, pos] for att in cross_atts]
    cross_atts = [att.mean(2) for att in cross_atts]

    # 4. interpolate these cross att maps to 64x64, and then apply weight
    cross_atts_64 = []
    cross_weight_sum = sum(config.cross_weight)
    for idx, att in enumerate(cross_atts):
        att = att.unsqueeze(0).unsqueeze(0)
        att = F.interpolate(att, size=(64, 64), mode="bilinear")
        att = att.squeeze() / att.max()
        cross_atts_64.append(att * config.cross_weight[idx] / cross_weight_sum)

    # 5. sum up and normalize these cross att maps
    cross_atts_64 = torch.stack(cross_atts_64).sum(0).view(64 * 64, 1)
    cross_atts_64 = F.sigmoid(config.norm_factor * (cross_atts_64 - config.norm_bias))
    cross_atts_64 = (cross_atts_64 - cross_atts_64.min()) / (
        cross_atts_64.max() - cross_atts_64.min()
    )

    return cross_atts_64


def aggregate_self_att(controller: AttentionStore):
    self_att_8 = [att for att in controller.attention_store["mid_self"]]
    self_att_16 = [att for att in controller.attention_store["up_self"][0:3]]
    self_att_32 = [att for att in controller.attention_store["up_self"][3:6]]
    self_att_64 = [att for att in controller.attention_store["up_self"][6:9]]

    weight_list = self_att_64 + self_att_32 + self_att_16 + self_att_8
    weight = [sqrt(weights.shape[-2]) for weights in weight_list]
    weight_sum = sum(weight)
    weight = [w / weight_sum for w in weight]
    aggre_weights = torch.zeros((64, 64, 64, 64)).to(self_att_64[0].device)
    for index, weights in enumerate(weight_list):
        size = round(sqrt(weights.shape[-1]))
        ratio = int(64 / size)
        weights = weights.reshape(-1, size, size).unsqueeze(0)
        weights = F.interpolate(weights, size=(64, 64), mode="bilinear")
        weights = weights.squeeze()
        weights = weights.reshape(size, size, 64, 64)
        weights = weights / torch.sum(weights, dim=(2, 3), keepdim=True)
        weights = weights.repeat_interleave(ratio, dim=0)
        weights = weights.repeat_interleave(ratio, dim=1)
        aggre_weights += weights * weight[index]
    return aggre_weights.view(64 * 64, 64 * 64)


def aggregate_self_64(controller: AttentionStore):
    self_att_64 = [att for att in controller.attention_store["up_self"][6:9]]
    return torch.stack(self_att_64).mean(0)
