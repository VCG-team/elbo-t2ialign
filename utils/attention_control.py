# Modified from Prompt-to-Prompt(ICLR 2023) https://github.com/google/prompt-to-prompt
import abc
from math import sqrt
from typing import Dict, List

import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image
from einops import rearrange
from omegaconf import DictConfig
from torch.distributions import Normal

T = torch.Tensor
TL = List[T]


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn: T, is_cross: bool, place_in_unet: str) -> T:
        raise NotImplementedError

    def __call__(self, attn: T, is_cross: bool, place_in_unet: str):
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

    def forward(self, attn: T, is_cross: bool, place_in_unet: str):
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

    def get_average_attention(self) -> Dict[str, TL]:
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
        self.step_store: Dict[str, TL] = self.get_empty_store()
        self.attention_store: Dict[str, TL] = {}


def register_attention_control(
    pipe: AutoPipelineForText2Image,
    controller: AttentionStore,
    config: DictConfig,
):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            # keep variable name with original code(prompt-to-prompt)
            x = hidden_states
            context = encoder_hidden_states

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

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1, dtype=sim.dtype)

            # in dds loss, we use batched input(batch size = 4)
            # input 0 is negative cond and source img, input 1 is negative cond and target img
            # input 2 is source cond and source img, input 3 is target cond and target img
            # see DDSLoss.get_eps_prediction in loss.py for more details
            with torch.inference_mode():
                if not is_cross:
                    source_att = attn[2 * h : 3 * h]
                    # save self attention of source img(take mean for all attention heads)
                    controller(source_att.mean(0), is_cross, place_in_unet)
                elif config.merge_type == "latent":
                    source_q, target_q = q[2 * h : 3 * h], q[3 * h :]
                    mix_q = source_q + config.target_factor * (source_q - target_q)
                    source_k = k[2 * h : 3 * h]
                    sim_mix = torch.einsum("b i d, b j d -> b i j", mix_q, source_k)
                    mix_att = (sim_mix * self.scale).softmax(
                        dim=-1, dtype=sim_mix.dtype
                    )
                    # save cross attention between source cond and mixed img(take mean for all attention heads)
                    controller(mix_att.mean(0), is_cross, place_in_unet)
                elif config.merge_type == "attention":
                    target_q = q[3 * h :]
                    source_k = k[2 * h : 3 * h]
                    sim_t = torch.einsum("b i d, b j d -> b i j", target_q, source_k)
                    target_att = (sim_t * self.scale).softmax(dim=-1, dtype=sim_t.dtype)
                    source_att = attn[2 * h : 3 * h]
                    mix_att = source_att + config.target_factor * (
                        source_att - target_att
                    )
                    controller(mix_att.mean(0), is_cross, place_in_unet)
                else:
                    raise ValueError(f"Invalid merge type: {config.merge_type}")

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            out = to_out(out)
            # if loss_type == "cds", we need to save self attention output
            if config.loss_type == "cds" and place_in_unet == "up" and not is_cross:
                self.self_out = out[2:].clone()
            return out

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
    sub_nets = pipe.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


@torch.inference_mode()
def aggregate_cross_att(
    att_maps: Dict[str, TL],
    pos: List[int],
    config: DictConfig,
) -> T:
    out, weight_sum, max_res = 0, 0, None
    cur_res, cur_res_idx, cur_layer = None, 0, 1

    if config.cross_gaussian_var != 0:
        down_cross_num, mid_cross_num, up_cross_num = (
            len(att_maps["down_cross"]),
            len(att_maps["mid_cross"]),
            len(att_maps["up_cross"]),
        )
        down_mid_sum = down_cross_num + mid_cross_num
        # calculate the scale_factor to scale the down and up to the same range
        scale_factor = down_cross_num / up_cross_num
        # set the mean of the gaussian distribution to the middle
        middle = down_cross_num + (mid_cross_num + 1) / 2
        weight_dist = Normal(middle, config.cross_gaussian_var)

    for location in ["down", "mid", "up"]:
        for att in att_maps[f"{location}_cross"]:  # attn shape: (res*res, prompt len)
            res = round(sqrt(att.shape[0]))
            if max_res is None:
                max_res, cur_res = res, res
            if res != cur_res:
                cur_res, cur_res_idx = res, cur_res_idx + 1
            att = att[:, pos].mean(1)  # get cross att maps for specific words
            att = att.reshape(1, 1, res, res)
            att = F.interpolate(att, size=(max_res, max_res), mode="bilinear")
            if config.cross_gaussian_var != 0:
                # use scaled cur_layer when in up_cross
                adjusted_layer = (
                    (cur_layer - down_mid_sum) * scale_factor + down_mid_sum
                    if cur_layer > down_mid_sum
                    else cur_layer
                )
                weight = weight_dist.log_prob(torch.tensor(adjusted_layer)).exp().item()
            else:
                weight = config.cross_weight[cur_res_idx]
            out += att * weight  # apply weight
            weight_sum += weight
            cur_layer += 1
    out /= weight_sum
    return out.view(max_res * max_res, 1)


@torch.inference_mode()
def aggregate_self_att(
    att_maps: Dict[str, TL],
    config: DictConfig,
) -> T:
    out, weight_sum, max_res = (
        [0] * len(config.self_weight),
        [0] * len(config.self_weight),
        None,
    )
    cur_res, cur_res_idx, cur_layer = None, 0, 1

    if len(config.self_gaussian_var) != 0:
        out = [0] * len(config.self_gaussian_var)
        weight_sum = [0] * len(config.self_gaussian_var)
        down_self_num, mid_self_num, up_self_num = (
            len(att_maps["down_self"]),
            len(att_maps["mid_self"]),
            len(att_maps["up_self"]),
        )
        down_mid_sum = down_self_num + mid_self_num
        # calculate the scale_factor to scale the down and up to the same range
        scale_factor = down_self_num / up_self_num
        # set the mean of the gaussian distribution to the middle
        middle = down_self_num + (mid_self_num + 1) / 2
        weight_dist = [Normal(0, var) for var in config.self_gaussian_var]

    for location in ["down", "mid", "up"]:
        for att in att_maps[f"{location}_self"]:  # attn shape: (res*res, res*res)
            res = round(sqrt(att.shape[0]))
            if max_res is None:
                max_res, cur_res = res, res
            if res != cur_res:
                cur_res, cur_res_idx = res, cur_res_idx + 1
            # refer to diffseg (CVPR 2024), we interpolate and then repeat (repeat is important)
            # related code: https://github.com/google/diffseg/blob/main/diffseg/segmentor.py#L40
            # paper: https://arxiv.org/abs/2308.12469 (Attention Aggregation Section)
            att = att.reshape(res, res, res, res)
            att = F.interpolate(att, size=(max_res, max_res), mode="bilinear")
            att = att.repeat_interleave(round(max_res / res), dim=0)
            att = att.repeat_interleave(round(max_res / res), dim=1)
            if len(config.self_gaussian_var) != 0:
                # use scaled cur_layer when in up_self
                adjusted_layer = (
                    (cur_layer - down_mid_sum) * scale_factor + down_mid_sum
                    if cur_layer > down_mid_sum
                    else cur_layer
                )
                adjusted_layer = (
                    adjusted_layer
                    if adjusted_layer <= middle
                    else 2 * middle - adjusted_layer
                )
                weight = [
                    weight_dist[i].log_prob(torch.tensor(adjusted_layer)).exp().item()
                    for i in range(len(config.self_gaussian_var))
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
