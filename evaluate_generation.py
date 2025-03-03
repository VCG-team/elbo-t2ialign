import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils.datasets import TextDataset
from utils.parse_args import parse_args

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    config = parse_args("evaluation")
    dataset = TextDataset(config)
    output_dir = config.output_path
    img_dir = os.path.join(output_dir, "images")

    # load model
    _ = torch.set_grad_enabled(False)
    clip_dtype = torch.float16 if config.clip.dtype == "fp16" else torch.float32
    clip_processor = CLIPProcessor.from_pretrained(
        config.clip.variant,
        cache_dir=config.model_dir,
    )
    clip_model = CLIPModel.from_pretrained(
        config.clip.variant,
        use_safetensors=True,
        cache_dir=config.model_dir,
        torch_dtype=clip_dtype,
        device_map=config.clip.device_map,
    )

    # compute clip scores
    clip_score_per_img = defaultdict(list)
    for img_file in tqdm(os.listdir(img_dir), desc="computing clip scores..."):
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        txt_idx = int(re.search(r"\d+", img_file.split("_")[0]).group())
        inputs = clip_processor(
            text=dataset.prompts[txt_idx],
            images=[img],
            return_tensors="pt",
            padding=True,
        )
        outputs = clip_model(**inputs)
        clip_score_per_img[txt_idx].append(outputs.logits_per_image.cpu().numpy())

    # compute average clip scores
    scores_sum = 0
    for txt_idx, scores in clip_score_per_img.items():
        scores_sum += np.mean(scores)
    avg_clip_score = float(scores_sum / len(clip_score_per_img))

    # save scores
    with open(os.path.join(output_dir, "evaluate_generation.json"), "w") as f:
        json.dump({"avg_clip_score": avg_clip_score}, f)
