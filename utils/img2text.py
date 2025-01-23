import base64
import hashlib
import io
import re
from os.path import join

import torch
from diskcache import Cache
from openai import OpenAI
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class Img2Text:
    def __init__(self, config):
        self.variant = config.img2text.variant
        self.is_local = config.img2text.api_url is None
        if self.is_local:
            self.dtype = (
                torch.float16 if config.img2text.dtype == "fp16" else torch.float32
            )
            self.processor = AutoProcessor.from_pretrained(
                self.variant, cache_dir=config.model_dir
            )
            model = AutoModelForVision2Seq.from_pretrained(
                self.variant,
                cache_dir=config.model_dir,
                torch_dtype=self.dtype,
                use_safetensors=True,
                device_map=config.img2text.device_map,
            )
            self.model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            cache_path = join(
                config.cache_dir,
                "img2text",
                f"{self.variant.replace('/','--')}--{config.img2text.dtype}",
            )
        else:
            self.api_key = config.img2text.api_key
            self.api_url = config.img2text.api_url
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            self.sys_prompt = config.img2text.system_prompt
            self.max_completion_tokens = config.img2text.max_completion_tokens
            clean_prompt = re.sub(r"\s+", " ", self.sys_prompt.strip())
            hash_prompt = hashlib.md5(clean_prompt.encode()).hexdigest()[0:16]
            cache_path = join(
                config.cache_dir,
                "img2text",
                f"{self.variant}--{hash_prompt}--{self.max_completion_tokens}",
            )
        self.cache = Cache(
            cache_path,
            cull_limit=0,
            eviction_policy="none",
        )

    def __call__(self, img: Image, img_name: str, prompt: str) -> str:
        clean_prompt = re.sub(r"\s+", " ", prompt.strip())
        hash_prompt = hashlib.md5(clean_prompt.encode()).hexdigest()[0:16]
        img_hash = hashlib.md5(img.tobytes()).hexdigest()[0:16]
        key = f"{img_name}&{hash_prompt}&{img_hash}"
        if key in self.cache:
            return self.cache[key]
        if self.is_local:
            text = self._get_text_locally(img, clean_prompt)
        else:
            text = self._get_text_with_api(img, clean_prompt)
        self.cache[key] = text
        return text

    # use vision model to generate text
    # details: https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForVision2Seq
    @torch.inference_mode()
    def _get_text_locally(self, img: Image, prompt: str) -> str:
        model_input = self.processor(images=img, text=prompt, return_tensors="pt")
        model_input = model_input.to(self.model.device)
        model_out = self.model.generate(**model_input)[0]
        text: str = self.processor.decode(
            model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if text.startswith(prompt):
            text = text[len(prompt) :]
        text = re.sub(r"\s+", " ", text.strip())
        return text

    # use OpenAI API to generate text
    # details: https://platform.openai.com/docs/api-reference/chat
    def _get_text_with_api(self, img: Image, prompt: str) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.variant,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.sys_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                },
            ],
            max_completion_tokens=self.max_completion_tokens,
        )
        return response.choices[0].message.content
