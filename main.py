import os
import shutil
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from datasets import VOC12Dataset
from utils.dds_utils import image_optimization
from utils.ptp_utils import AttentionStore, generate_att_v2, register_attention_control

CATEGORY = [
    "plane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "buses",
    "car",
    "cat",
    "chair",
    "cow",
    "table",
    "dog",
    "horse",
    "motorbike",
    "people",
    "plant",
    "sheep",
    "sofa",
    "train",
    "monitor",
]

BACKGROUND_CATEGORY = [
    "ground",
    "land",
    "grass",
    "tree",
    "building",
    "wall",
    "sky",
    "lake",
    "water",
    "river",
    "sea",
    "railway",
    "railroad",
    "keyboard",
    "helmet",
    "cloud",
    "house",
    "mountain",
    "ocean",
    "road",
    "rock",
    "street",
    "valley",
    "bridge",
    "sign",
]


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


def get_weight_rato(weight_list):
    sizes = []
    for weights in weight_list:
        sizes.append(np.sqrt(weights.shape[-2]))
    denom = np.sum(sizes)
    return sizes / denom


def aggregate_self_att(controller):
    self_att_8 = [att for att in controller.attention_store["mid_self"]]
    self_att_16 = [att for att in controller.attention_store["up_self"][0:3]]
    self_att_32 = [att for att in controller.attention_store["up_self"][3:6]]
    self_att_64 = [att for att in controller.attention_store["up_self"][6:9]]

    weight_list = self_att_64 + self_att_32 + self_att_16 + self_att_8
    weight = get_weight_rato(weight_list)
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


def get_mul_self_cross(cross_att, self_att):
    res = cross_att.shape[0]
    cross_att = cross_att.view(res * res, 1)
    self_cross = torch.matmul(self_att, cross_att)
    self_cross = self_cross.view(res, res)
    cross_att = cross_att.view(res, res)
    return self_cross


# srcfile 需要复制、移动的文件
# dstpath 目的地址
def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, f"{dstpath}/{fname}")  # 复制文件
        print("copy %s -> %s" % (srcfile, f"{dstpath}/{fname}"))


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    output_path = f"./output/test"
    img_output_path = f"{output_path}/images"
    code_output_path = f"{output_path}/codes"
    diffusion_path = "/data/wjl/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/c9ab35ff5f2c362e9e22fbafe278077e196057f0"
    blip_path = "/data/wjl/ptp_diffusion/blip/blip-image-captioning-large"
    device = torch.device("cuda:4")
    use_blip = True  # 是否使用Blip补全语句
    print_prompt = False  # 是否打印prompt
    self_times = 1  # 乘以self_aggre的次数
    cross_threshold = 0.4  # cross的二极分化阈值
    times = [1, 25, 50, 75, 100, 125, 150]
    same_seeds(4307)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(img_output_path, exist_ok=True)

    mycopyfile(os.path.abspath(__file__), code_output_path)
    mycopyfile("./utils/ptp_utils.py", code_output_path)

    dataset = VOC12Dataset("./data/voc12_train_id.txt", "./VOCdevkit/VOC2012")
    print(f"data_size: {len(dataset)}")

    pipeline = StableDiffusionPipeline.from_pretrained(
        diffusion_path, local_files_only=True
    ).to(device)
    controller = AttentionStore()
    register_attention_control(pipeline, controller)

    if use_blip:
        blip_processor = BlipProcessor.from_pretrained(blip_path)
        blip_model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)

    for k, (img, label) in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing images..."
    ):

        images = []
        h, w = img.size
        y = torch.where(label)[0]
        img_512 = np.array(img.resize((512, 512), resample=Image.BILINEAR))

        for i in range(y.shape[0]):
            text_source = f"a photograph of {CATEGORY[y[i].item()]}."
            text_target = f"a photograph of ''."

            if use_blip:
                with torch.no_grad():
                    blip_inputs = blip_processor(
                        img, text_source[:-1], return_tensors="pt"
                    ).to(device)
                    blip_out = blip_model.generate(**blip_inputs)
                    blip_out_prompt = blip_processor.decode(
                        blip_out[0], skip_special_tokens=True
                    )
                    length = len(text_source[:-1])

                    text_source = (
                        text_source[:-1]
                        + "++"
                        + blip_out_prompt[length:]
                        + " and "
                        + ",".join(BACKGROUND_CATEGORY)
                        + "."
                    )
                    text_target = text_target[:-1] + "."
            if print_prompt:
                tqdm.write(f"image: {k}, source_text: {text_source}")
                tqdm.write(f"image: {k}, target_text: {text_target}")

            grad_list_dds = None

            controller.reset()
            image_optimization(
                pipeline,
                img_512,
                text_source,
                text_target,
                use_dds=True,
                num_iters=10,
                grad_list=grad_list_dds,
                device=device,
                times=times,
            )

            cross_att_map = generate_att_v2(
                [text_source],
                controller,
                4,
                weight=[0.3, 0.5, 0.1, 0.1],
                cross_threshold=cross_threshold,
            )
            self_att = aggregate_self_att(controller).view(64 * 64, 64 * 64)

            for _ in range(self_times):
                cross_att_map = torch.matmul(self_att, cross_att_map)

            self_64 = (
                torch.stack([att for att in controller.attention_store["up_self"][6:9]])
                .mean(0)
                .cpu()
            )
            att_map = torch.matmul(self_64, cross_att_map).view(64, 64)

            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
            att_map = F.sigmoid(8 * (att_map - 0.4))
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

            images.append((att_map).unsqueeze(0).repeat(3, 1, 1))

        images = torch.stack(images)
        images = F.interpolate(
            images, size=(h, w), mode="bilinear", align_corners=False
        )

        for i in range(images.shape[0]):
            images[i] = (
                (images[i] - images[i].min()) / (images[i].max() - images[i].min())
            ) * 255

        for i in range(0, y.shape[0]):
            class_emb = CATEGORY[y[i].item()]
            cv2.imwrite(
                f"{img_output_path}/{k}_{class_emb}.png",
                images[i].permute(1, 2, 0).cpu().numpy(),
            )
