import json
import os
import warnings
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import spacy
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
)

from datasets import build_dataset

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("--dataset-cfg", type=str, default="./configs/dataset/voc.yaml")
    parser.add_argument("--io-cfg", type=str, default="./configs/io/io.yaml")
    parser.add_argument(
        "--method-cfg", type=str, default="./configs/method/classification.yaml"
    )
    args, unknown = parser.parse_known_args()

    dataset_cfg = OmegaConf.load(args.dataset_cfg)
    io_cfg = OmegaConf.load(args.io_cfg)
    method_cfg = OmegaConf.load(args.method_cfg)
    cli_cfg = OmegaConf.from_dotlist(unknown)

    config = OmegaConf.merge(dataset_cfg, io_cfg, method_cfg, cli_cfg)
    config.output_path = config.output_path[config.dataset]
    os.makedirs(config.output_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_path, "classification.yaml"))

    # load model
    _ = torch.set_grad_enabled(False)
    nlp = spacy.load("en_core_web_sm")

    clip_device = config.clip.device
    clip_tokenizer = CLIPTokenizer.from_pretrained(config.clip.path)
    clip_processor = CLIPImageProcessor.from_pretrained(config.clip.path)
    clip_model = CLIPModel.from_pretrained(config.clip.path).to(clip_device)

    blip_device = config.blip.device
    blip_processor = BlipProcessor.from_pretrained(config.blip.path)
    blip_model = BlipForConditionalGeneration.from_pretrained(config.blip.path).to(
        blip_device
    )

    # load image dataset for classification
    dataset = build_dataset(config)

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
        labels_input = labels_input.to(clip_device)
        labels_features = clip_model.get_text_features(**labels_input)
        labels_features = labels_features / labels_features.norm(
            p=2, dim=-1, keepdim=True
        )

    # prepare sentences' embeddings to compute cos similarity of image and text
    if config.enable_text_and_image_similarity:
        sentences = [f"a photograph of {label}" for label in labels]
        sentences_input = clip_tokenizer(sentences, padding=True, return_tensors="pt")
        sentences_input = sentences_input.to(clip_device)
        sentences_features = clip_model.get_text_features(**sentences_input)
        sentences_features = sentences_features / sentences_features.norm(
            p=2, dim=-1, keepdim=True
        )

    # generate multi-label classification results
    predict = defaultdict(dict)
    text_prompts = [
        "",
        "a photograph of",
        "an image of",
        "there are many things in the image, including",
        "except for the most prominent objects, there are",
    ]

    image_classification_bar = tqdm(dataset)
    image_classification_bar.set_description("image multi-label classification")
    for data_idx, (image, label, name) in enumerate(image_classification_bar):
        label_predict = set()

        # 1. generate text for all text prompts, and extract nouns and all words
        # TODO: cache
        noun, all_word = set(), set()
        for prompt in text_prompts:
            blip_inputs = blip_processor(image, prompt, return_tensors="pt")
            blip_inputs = blip_inputs.to(blip_device)
            blip_out = blip_model.generate(**blip_inputs)
            text = blip_processor.decode(blip_out[0], skip_special_tokens=True)
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
            image_input = image_input.to(clip_device)
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
            noun_input = noun_input.to(clip_device)
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
