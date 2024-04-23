import os
import warnings
from collections import defaultdict

import numpy as np
import spacy
import torch
import tqdm
from omegaconf import OmegaConf
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizer,
)

from datasets import build_dataset


def analyse_text(text):
    # analyse sentence with spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    noun = []
    all_word = []
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            noun.append(token.lemma_)
            all_word.append(token.lemma_)
        else:
            all_word.append(token.text)
    return noun, all_word


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    config_path = "./configs/voc12/classification.yaml"
    config = OmegaConf.load(config_path)

    # load model
    clip_device = config.clip_device
    clip_tokenizer = CLIPTokenizer.from_pretrained(config.clip_path)
    clip_image_processor = CLIPImageProcessor.from_pretrained(config.clip_path)
    clip_model = CLIPModel.from_pretrained(config.clip_path).to(clip_device)
    clip_text_model = clip_model.text_model

    blip_device = config.blip_device
    blip_processor = BlipProcessor.from_pretrained(config.blip_path)
    blip_model = BlipForConditionalGeneration.from_pretrained(config.blip_path).to(
        blip_device
    )

    # load image dataset for classification
    dataset_train = build_dataset(config)

    # load labels and synonym for different dataset
    labels_synonym = config.labels_synonym
    labels = list(labels_synonym.keys())
    num_cls = len(labels)

    # prepare labels' embeddings to compute cos similarity of text
    if config.enable_text_similarity:
        labels_features = torch.zeros((1, 768)).to(clip_device)
        label_embedding_bar = tqdm.tqdm(labels)
        label_embedding_bar.set_description("label embedding")
        for label in label_embedding_bar:
            label_input = clip_tokenizer(
                label,
                padding="max_length",
                max_length=15,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding_len = len(clip_tokenizer.encode(label)) - 2
            tmp_features = clip_text_model(label_input.input_ids.to(clip_device))[0][
                :, 1 : text_embedding_len + 1, :
            ].mean(1)
            labels_features = torch.cat([labels_features, tmp_features], 0)
        labels_features = labels_features[1:, :]
        labels_features = labels_features / torch.norm(
            labels_features, dim=-1, keepdim=True
        )

    # prepare sentences' embeddings to compute cos similarity of image and text
    if config.enable_text_and_image_similarity:
        sentences = ["A photo of a " + label for label in labels]
        sentences_input = clip_tokenizer(
            sentences, padding=True, return_tensors="pt"
        ).to(clip_device)
        sentences_features = clip_model.get_text_features(**sentences_input)
        sentences_features = sentences_features / sentences_features.norm(
            p=2, dim=-1, keepdim=True
        )

    # generate multi-label classification results
    predict = defaultdict(dict)
    text_prompts = ["", "A picture of"]

    image_classification_bar = tqdm.tqdm(dataset_train)
    image_classification_bar.set_description("image multi-label classification")
    for data_idx, (image, label, name) in enumerate(image_classification_bar):

        # 1. get image, ground truth label, image path from data
        gt_label = label.tolist()
        tmp_pre = []

        # 2. generate text for all text prompts, and extract nouns and all words
        noun, all_word, text_list = [], [], []
        for text in text_prompts:
            inputs = blip_processor(image, text, return_tensors="pt").to(blip_device)
            out = blip_model.generate(**inputs)
            texts = blip_processor.decode(out[0], skip_special_tokens=True)
            noun_tmp, all_word_tmp = analyse_text(texts)
            noun.extend(noun_tmp)
            all_word.extend(all_word_tmp)
            text_list.append(texts)
        noun = list(set(noun))
        all_word = list(set(all_word))

        # 3. synonym matching to predict label
        if config.enable_synonym_matching:
            tmp_list = []
            for c in labels:
                if any(non in all_word for non in labels_synonym[c]):
                    tmp_list.append(labels.index(c))
            tmp_pre.append(set(tmp_list))

        # 4. cos similariry of image and text to predict label
        if config.enable_text_and_image_similarity:
            image_input = clip_image_processor(images=image, return_tensors="pt").to(
                clip_device
            )
            image_features = clip_model.get_image_features(**image_input)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            logit_scale = clip_model.logit_scale.exp()
            logits_per_text = (
                torch.matmul(sentences_features, image_features.t()) * logit_scale
            )
            logits_per_image = logits_per_text.t()[0]
            logits_max = logits_per_image.max()
            x = torch.where(
                logits_per_image >= logits_max * config.text_and_image_threshold
            )
            tmp_pre.append(set(x[0].tolist()))

        # 5. cos similarity of text embdding to predict label
        if config.enable_text_similarity:
            text_embedding_len = [len(clip_tokenizer.encode(n)) - 2 for n in noun]
            text_embedding_len = torch.tensor(text_embedding_len).to(clip_device)
            noun_input = clip_tokenizer(
                noun,
                padding="max_length",
                max_length=15,
                truncation=True,
                return_tensors="pt",
            )
            tmp_features = clip_text_model(noun_input.input_ids.to(clip_device))[0]
            noun_features = torch.zeros((1, 768)).to(clip_device)
            for i in range(0, tmp_features.shape[0]):
                noun_features = torch.cat(
                    [
                        noun_features,
                        tmp_features[i, 1 : text_embedding_len[i] + 1, :]
                        .mean(0)
                        .unsqueeze(0),
                    ],
                    0,
                )
            noun_features = noun_features[1:, :]
            noun_features = noun_features / torch.norm(
                noun_features, dim=-1, keepdim=True
            )
            similarity = torch.mm(labels_features, noun_features.T)
            x, y = torch.where(similarity > config.text_threshold)

            tmp_pre.append(set(x.tolist()))

        tmp_pre = set.union(*tmp_pre)
        tmp_pre_onehot = np.zeros(num_cls)
        tmp_pre_onehot[list(tmp_pre)] = 1
        predict[name] = tmp_pre_onehot

    os.makedirs(config.output_path, exist_ok=True)
    output_path = os.path.join(config.output_path, "predict_cls.npy")
    np.save(output_path, predict, allow_pickle=True)
