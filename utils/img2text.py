import hashlib
import re
from os.path import join

import torch
from diskcache import Cache
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class Img2Text:
    def __init__(self, config: DictConfig):
        self.variant = config.img2text.variant
        self.is_local = config.img2text.api_url is None
        if self.is_local:
            self.dtype = (
                torch.float16 if config.img2text.dtype == "fp16" else torch.float32
            )
            self.device = torch.device(config.img2text.device)
            self.processor = AutoProcessor.from_pretrained(
                self.variant, cache_dir=config.model_dir
            )
            model = AutoModelForVision2Seq.from_pretrained(
                self.variant,
                cache_dir=config.model_dir,
                torch_dtype=self.dtype,
                use_safetensors=True,
            ).to(self.device)
            self.model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            cache_path = join(
                config.cache_dir,
                "img2text",
                f"{self.variant.replace('/','--')}--{config.img2text.dtype}",
            )
        else:
            self.api_key = config.img2text.api_key
            self.api_url = config.img2text.api_url
            cache_path = join(config.cache_dir, "img2text", self.variant)
        self.cache = Cache(
            cache_path,
            cull_limit=0,
            eviction_policy="none",
        )

    def __call__(self, img: Image, img_name: str, prompt: str) -> str:  # todo space
        clean_prompt = re.sub(r"\s+", " ", prompt.strip())
        hash_prompt = hashlib.md5(clean_prompt.encode()).hexdigest()
        key = f"{img_name}--{hash_prompt}"
        if key in self.cache:
            return self.cache[key]
        if self.is_local:
            text = self._get_text_locally(img, clean_prompt)
        else:
            text = self._get_text_with_api(img, clean_prompt)
        self.cache[key] = text
        return text

    @torch.inference_mode()
    def _get_text_locally(self, img: Image, prompt: str) -> str:
        model_input = self.processor(images=img, text=prompt, return_tensors="pt")
        model_input = model_input.to(self.device)
        model_out = self.model.generate(**model_input)[0]
        text: str = self.processor.decode(
            model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if text.startswith(prompt):
            text = text[len(prompt) :]
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _get_text_with_api(self, img: Image, prompt: str) -> str:
        pass
