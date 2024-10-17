# Modified from DDS(ICCV 2023) https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

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
        loss_type: str = "dds",
    ):
        if DDSLoss.init_flag:
            return
        DDSLoss.init_flag = True
        self.loss_type = loss_type

    def get_loss(
        self,
        z_target: T,
        eps_pred_source: T,
        eps_pred_target: T,
        eps: T,
    ) -> TS:
        if self.loss_type == "sds":
            grad = eps_pred_target - eps
        elif self.loss_type == "dds" or self.loss_type == "cds":
            grad = eps_pred_target - eps_pred_source
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        loss = z_target * grad.clone().detach()
        loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss
