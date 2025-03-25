import json
import os
import warnings
from collections import defaultdict

import numpy as np
import spacy
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

from utils.datasets import SegDataset
from utils.img2text import Img2Text
from utils.parse_args import parse_args

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    config = parse_args("run_classification")

    # load model
    _ = torch.set_grad_enabled(False)
    nlp = spacy.load("en_core_web_sm")

    clip_dtype = torch.float16 if config.clip.dtype == "fp16" else torch.float32
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        config.clip.variant, cache_dir=config.model_dir
    )
    clip_processor = CLIPImageProcessor.from_pretrained(
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
    clip_model = torch.compile(clip_model, mode="reduce-overhead", fullgraph=True)

    img2text = Img2Text(config)

    # load image dataset for classification
    dataset = SegDataset(config)

    # load labels and synonym for different dataset
    labels_synonym = config.category
    labels = list(labels_synonym.keys())
    num_cls = len(labels)

    # prepare labels' embeddings to compute cos similarity of text
    if config.enable_text_similarity:
        # use clip model to get text/image features
        # related code: https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/clip/modeling_clip.py#L932
        # related docs: https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel
        labels_input = clip_tokenizer(labels, padding=True, return_tensors="pt")
        labels_input = labels_input.to(clip_model.device)
        labels_features = clip_model.get_text_features(**labels_input)
        labels_features = labels_features / labels_features.norm(
            p=2, dim=-1, keepdim=True
        )

    # prepare sentences' embeddings to compute cos similarity of image and text
    if config.enable_text_and_image_similarity:
        sentences = [f"a photo of {label}" for label in labels]
        sentences_input = clip_tokenizer(sentences, padding=True, return_tensors="pt")
        sentences_input = sentences_input.to(clip_model.device)
        sentences_features = clip_model.get_text_features(**sentences_input)
        sentences_features = sentences_features / sentences_features.norm(
            p=2, dim=-1, keepdim=True
        )

    # generate multi-label classification results
    predict = defaultdict(dict)
    text_prompts = [
        "",
        "a photo of",
        "an image of",
        "there are many things in the image, including",
        "except for the most prominent objects, there are",
    ]

    for data_idx, (name, img_path, _, _) in enumerate(
        tqdm(dataset, desc=f"image multi-label classification of {dataset.name}")
    ):
        label_predict = set()
        image = Image.open(img_path).convert("RGB")

        # 1. generate text for all text prompts, and extract nouns and all words
        # TODO: cache
        noun, all_word = set(), set()
        for prompt in text_prompts:
            out_text = img2text(image, name, prompt)
            text = prompt + " " + out_text
            # use spacy to extract noun and all words, exclude prompt
            # related docs: https://spacy.io/usage/linguistic-features#pos-tagging
            for token in nlp(text)[len(nlp(prompt)) :]:
                if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    noun.add(token.lemma_)
                    all_word.add(token.lemma_)
                else:
                    all_word.add(token.text)

        # 2. synonym matching to predict label
        if config.enable_synonym_matching:
            for idx, c in enumerate(labels):
                for synonym in labels_synonym[c] + [c]:
                    if synonym in all_word:
                        label_predict.add(idx)
                        noun.discard(synonym)

        # 3. cos similariry of image and text to predict label
        if config.enable_text_and_image_similarity:
            image_input = clip_processor(images=image, return_tensors="pt")
            image_input = image_input.to(clip_model.device)
            image_features = clip_model.get_image_features(**image_input)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            logits = torch.matmul(sentences_features, image_features.t())
            x, _ = torch.where(logits >= config.text_and_image_threshold.absolute)
            label_predict.update(x.tolist())
            x, _ = torch.where(
                logits >= logits.max() * config.text_and_image_threshold.relative
            )
            label_predict.update(x.tolist())

        # 4. cos similarity of text embdding to predict label
        if config.enable_text_similarity and noun:
            noun_input = clip_tokenizer(list(noun), padding=True, return_tensors="pt")
            noun_input = noun_input.to(clip_model.device)
            noun_features = clip_model.get_text_features(**noun_input)
            noun_features = noun_features / noun_features.norm(
                p=2, dim=-1, keepdim=True
            )
            logits = torch.matmul(labels_features, noun_features.t())
            x, _ = torch.where(logits >= config.text_threshold)
            label_predict.update(x.tolist())

        label_predict_onehot = np.zeros(num_cls)
        label_predict_onehot[list(label_predict)] = 1
        predict[name] = label_predict_onehot

    # evaluate multi-label classification results
    gt_list = dataset.label_list
    predict_list = list(predict.values())
    # gt_list and predict_list have the same order, so we can directly compare them
    # see https://stackoverflow.com/a/47849121/12389770 for more details
    # for multi-label classification, we can use metrics in scikit-learn
    # related dics: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    metrics = {
        "micro f1": f1_score(gt_list, predict_list, average="micro"),
        "micro precision": precision_score(gt_list, predict_list, average="micro"),
        "micro recall": recall_score(gt_list, predict_list, average="micro"),
        "macro f1": f1_score(gt_list, predict_list, average="macro"),
        "macro precision": precision_score(gt_list, predict_list, average="macro"),
        "macro recall": recall_score(gt_list, predict_list, average="macro"),
        "class f1": f1_score(gt_list, predict_list, average=None).tolist(),
        "class precision": precision_score(
            gt_list, predict_list, average=None
        ).tolist(),
        "class recall": recall_score(gt_list, predict_list, average=None).tolist(),
    }

    # save multi-label classification results
    predict_output_path = os.path.join(config.output_path, "cls_predict.npy")
    np.save(predict_output_path, predict, allow_pickle=True)
    metrics_output_path = os.path.join(
        config.output_path, "classification_metrics.json"
    )
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
