import os
import random
import shutil
import warnings
from typing import List, Optional

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

from utils.datasets import TextDataset
from utils.diffusion import Diffusion
from utils.parse_args import parse_args

T = torch.Tensor
TL = List[T]
TN = Optional[T]

nlp = spacy.load("en_core_web_sm")
NON_ENTITY_KEYWORDS = set()


def extract_filtered_noun_phrases(text: str) -> List[str]:
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        root_word = chunk.root.text.lower()
        if chunk.root.pos_ != "NOUN" or root_word in NON_ENTITY_KEYWORDS:
            continue
        noun_phrases.append(chunk.text)
    return noun_phrases


def parse_timesteps(timesteps: List[List]) -> List:
    parsed_timesteps = []
    for t in timesteps:
        if len(t) == 3:
            parsed_timesteps.extend(range(t[0], t[1], t[2]))
        else:  # len(t) == 4, random sample
            parsed_timesteps.extend([random.randint(t[1], t[2]) for _ in range(t[3])])
    return parsed_timesteps


def generate_prompt(prompt: str, phrases: List[str], weights: List) -> str:
    annotated_phrases = []
    for phrase, weight in zip(phrases, weights):
        start_idx = prompt.find(phrase)
        annotated_phrases.append((start_idx, start_idx + len(phrase), phrase, weight))

    annotated_phrases.sort(key=lambda x: x[0])

    result = prompt
    for start, end, phrase, weight in reversed(annotated_phrases):
        result = result[:start] + f"({phrase}){weight}" + result[end:]

    return result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = parse_args("generation")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_out_path = os.path.join(config.output_path, "images")
    if os.path.exists(img_out_path):
        shutil.rmtree(img_out_path)
    os.makedirs(img_out_path, exist_ok=True)
    dataset = TextDataset(config)
    data_len = min(len(dataset), config.num_prompts)
    NON_ENTITY_KEYWORDS = set(config.non_entity_keywords)

    diffusion_dtype = (
        torch.float16 if config.diffusion.dtype == "fp16" else torch.float32
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        config.diffusion.variant,
        torch_dtype=diffusion_dtype,
        use_safetensors=config.diffusion.use_safetensors,
        cache_dir=config.model_dir,
        device_map=config.diffusion.device_map,
    )
    diffusion = Diffusion(pipe)

    with torch.inference_mode():
        # 1. prepare timesteps
        train_timesteps = diffusion.scheduler.config.num_train_timesteps
        step_ratio = train_timesteps // config.infer_timesteps
        timesteps = list(range(train_timesteps - 1, 0, -step_ratio))

        # 2. prepare null text embedding
        negtive_prompt = ""
        neg_text_emb = diffusion.encode_prompt(negtive_prompt)

        # 3. generate image for each text prompt
        for txt_idx, text in enumerate(
            tqdm(
                dataset.prompts[:data_len],
                desc=f"generating images of {config.dataset}...",
            )
        ):
            # 3.1 prepare text embedding
            phrases = extract_filtered_noun_phrases(text)
            phrases_emb = [diffusion.encode_prompt(phrase) for phrase in phrases]
            text_emb = diffusion.encode_prompt(text)

            # 3.2 generate multiple images for each prompt
            for img_idx in range(config.num_imgs_per_prompt):
                # 3.2.1 prepare weight and latent
                weights = [1.0] * len(phrases)
                latent = diffusion.prepare_latent()
                # 3.2.2 reverse diffusion process
                for t_idx, t in enumerate(timesteps):
                    # get unweight model prediction for whole sentence
                    model_pred_all = diffusion.get_model_prediction(
                        [latent], [t], [text_emb]
                    )
                    # get unweight elbo and compute alignment score
                    if t_idx >= config.elbo_timesteps:
                        weights = [1.0] * len(phrases)
                    elif len(phrases_emb) > 1:
                        elbo = []
                        for p in phrases_emb:
                            model_pred = diffusion.get_model_prediction(
                                [latent], [t], [p]
                            )
                            elbo.append(
                                F.mse_loss(model_pred_all, model_pred, reduction="mean")
                            )
                        elbo = torch.stack(elbo)
                        elbo = (elbo - elbo.min()) / (elbo.max() - elbo.min())
                        weights = torch.round(
                            torch.pow(config.elbo_strength, elbo), decimals=2
                        ).tolist()
                    # generate weight prompt and get weight model prediction
                    weight_prompt = generate_prompt(text, phrases, weights)
                    weight_prompt_emb = diffusion.encode_prompt(weight_prompt)
                    model_pred_cond, model_pred_uncond = diffusion.get_model_prediction(
                        [latent, latent], [t, t], [weight_prompt_emb, neg_text_emb]
                    ).chunk(2)
                    model_pred = diffusion.classifier_free_guidance(
                        model_pred_uncond, model_pred_cond
                    )
                    # update latent with model prediction
                    latent = diffusion.step(
                        latent, t, max(0, t - step_ratio), model_pred
                    )
                # 3.2.3 save image
                img = diffusion.decode_latent(latent)[0]
                img.save(os.path.join(img_out_path, f"{txt_idx}_{img_idx}.png"))
