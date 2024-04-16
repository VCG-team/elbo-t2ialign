# Modified from Prompt-to-Prompt(ICLR 2023) https://github.com/google/prompt-to-prompt
import abc
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


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
        if attn.shape[1] <= 64**2:  # avoid memory overhead
            self.step_store[key].append(attn[16:24].mean(0))
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


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **cross_attention_kwargs,
        ):
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

            if is_cross:
                x[2] = x[2] + 0.5 * (x[3] - x[2])
                q = self.to_q(x)
                k = self.to_k(context)
                v = self.to_v(context)
                q, k, v = map(
                    lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
                )
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn_c = sim.softmax(dim=-1)

                attn_c = controller(attn_c, is_cross, place_in_unet)
            else:
                attn = controller(attn, is_cross, place_in_unet)

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
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def aggregate_attention(
    attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool
):

    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[0] == num_pixels:
                cross_maps = item.reshape(res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.stack(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def aggregate_all_attention(
    prompts,
    attention_store: AttentionStore,
    from_where: List[str],
    is_cross: bool,
    select: int,
):
    attention_maps = attention_store.get_average_attention()
    att_8 = []
    att_16 = []
    att_32 = []
    att_64 = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[0] == 8 * 8:
                cross_maps = item.reshape(len(prompts), 8, 8, item.shape[-1])[select]
                att_8.append(cross_maps)
            if item.shape[0] == 16 * 16:
                cross_maps = item.reshape(len(prompts), 16, 16, item.shape[-1])[select]
                att_16.append(cross_maps)
            if item.shape[0] == 32 * 32:
                cross_maps = item.reshape(len(prompts), 32, 32, item.shape[-1])[select]
                att_32.append(cross_maps)
            if item.shape[0] == 64 * 64:
                cross_maps = item.reshape(len(prompts), 64, 64, item.shape[-1])[select]
                att_64.append(cross_maps)
    atts = []
    for att in [att_8, att_16, att_32, att_64]:
        att = torch.stack(att, dim=0)
        att = att.sum(0) / att.shape[0]
        atts.append(att.cpu())
    return atts


def aggregate_self_att(controller: AttentionStore):
    self_att_8 = [att for att in controller.attention_store["mid_self"]]
    self_att_16 = [att for att in controller.attention_store["up_self"][0:3]]
    self_att_32 = [att for att in controller.attention_store["up_self"][3:6]]
    self_att_64 = [att for att in controller.attention_store["up_self"][6:9]]

    weight_list = self_att_64 + self_att_32 + self_att_16 + self_att_8
    weight = [np.sqrt(weights.shape[-2]) for weights in weight_list]
    weight = weight / np.sum(weight)
    aggre_weights = torch.zeros((64, 64, 64, 64)).to(self_att_64[0].device)
    for index, weights in enumerate(weight_list):
        size = int(np.sqrt(weights.shape[-1]))
        ratio = int(64 / size)
        weights = weights.reshape(-1, size, size)
        weights = (
            F.interpolate(
                weights.unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .squeeze()
        )
        weights = weights.reshape(size, size, 64, 64)
        weights = weights / torch.sum(weights, dim=(2, 3), keepdim=True)
        weights = weights.repeat_interleave(ratio, dim=0)
        weights = weights.repeat_interleave(ratio, dim=1)
        aggre_weights += weights * weight[index]
    return aggre_weights.cpu()


def generate_att(
    prompts,
    controller: AttentionStore,
    pos,
    is_self=True,
    is_multi_self=False,
    is_cross_norm=True,
    weight=[0.3, 0.5, 0.1, 0.1],
):
    layers = ("mid", "up", "down")
    cross_attention_maps = aggregate_all_attention(prompts, controller, layers, True, 0)
    self_attention_maps = aggregate_all_attention(prompts, controller, layers, False, 0)

    imgs = []
    for idx, res in enumerate([8, 16, 32, 64]):
        out_att = cross_attention_maps[idx].permute(2, 0, 1).float()
        if is_cross_norm:
            att_max = torch.amax(out_att, dim=(1, 2), keepdim=True)
            att_min = torch.amin(out_att, dim=(1, 2), keepdim=True)
            out_att = (out_att - att_min) / (att_max - att_min)
        if is_multi_self:
            self_att = self_attention_maps[idx].view(res * res, res * res).float()
            self_att = self_att / self_att.max()
            out_att = torch.matmul(
                self_att.unsqueeze(0), out_att.view(-1, res * res, 1)
            ).view(-1, res, res)
        if res != 64:
            out_att = F.interpolate(
                out_att.unsqueeze(0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        imgs.append(out_att * weight[idx])

    cross_att_map = torch.stack(imgs).sum(0)[pos].mean(0).view(64 * 64, 1)

    if is_self and not is_multi_self:
        self_att = self_attention_maps[3].view(64 * 64, 64 * 64).float()
        self_att = self_att / self_att.max()
        for _ in range(1):
            cross_att_map = torch.matmul(self_att, cross_att_map)

    att_map = cross_att_map.view(res, res)
    return att_map


def generate_att_v2(
    prompts,
    controller: AttentionStore,
    pos,
    weight=[0.3, 0.5, 0.1, 0.1],
    cross_threshold=0.4,
):
    layers = ["down", "mid", "up"]
    cross_attention_maps = aggregate_all_attention(prompts, controller, layers, True, 0)
    # self_attention_maps = aggregate_all_attention(prompts,controller, layers, False, 0)
    out_atts = []
    for idx, res in enumerate([8, 16, 32, 64]):
        next_word = prompts[0].split(" ")[pos]
        if next_word.endswith("ing"):
            cross_att = (
                cross_attention_maps[idx][:, :, [pos, pos + 1]]
                .mean(2)
                .view(res, res)
                .float()
            )
        else:
            cross_att = (
                cross_attention_maps[idx][:, :, [pos]].mean(2).view(res, res).float()
            )
        if res != 64:
            cross_att = (
                F.interpolate(
                    cross_att.unsqueeze(0).unsqueeze(0),
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .squeeze()
            )
        cross_att = cross_att / cross_att.max()
        out_atts.append(cross_att * weight[idx])

    cross_att_map = torch.stack(out_atts).sum(0).view(64 * 64, 1)
    cross_att_map = F.sigmoid(8 * (cross_att_map - cross_threshold))
    cross_att_map = (cross_att_map - cross_att_map.min()) / (
        cross_att_map.max() - cross_att_map.min()
    )

    return cross_att_map
