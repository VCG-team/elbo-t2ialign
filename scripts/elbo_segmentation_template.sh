#!/bin/bash
# find least used gpu, min memory first, min utilization second
# set device to "" to use all gpus, or set device to specific gpu indexes to use them, e.g. "0,1,2"
device=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | sed '1d' | sort -t ',' -k 2,2n -k 3,3n | head -n 1 | cut -d ',' -f 1)

# get script name as output folder name(remove file extension)
output_folder=$(basename $0 | rev | cut -d '.' -f 2- | rev)
output_path="./output/${output_folder}"

# experiment datasets
datasets=("voc_sim" "coco_cap" "voc" "context" "coco" "ade20k")
# which dataset variant to use, supported:
# 1. "": original dataset
# 2. "_100": small dataset(100 images) randomly selected from original dataset
# 3. "_10_car": small dataset(10 images) randomly selected from original dataset with car class
dataset_suffix=""

# when using small dataset, we recommend saving intermediate results for visualization
if [ "${dataset_suffix}" = "" ]; then
    save_mask=False
    save_cross_att=False
else
    save_mask=True
    save_cross_att=True
fi
# optionally using classification.py to get class labels
use_cls_predict=False

# classification.py arguments using here document, see full list of options in ./config/method/classification.yaml
classification_args=$(cat << EOS
EOS
)

# segmentation_with_elbo.py arguments using here document, see full list of options in ./config/method/segmentation.yaml
# clip.variant options: openai/clip-vit-large-patch14
# img2text.variant options: Salesforce/blip-image-captioning-large, Salesforce/blip2-opt-2.7b
# diffusion.variant options: stable-diffusion-v1-5/stable-diffusion-v1-5, CompVis/stable-diffusion-v1-4, stabilityai/sdxl-turbo, stabilityai/sd-turbo, stabilityai/stable-diffusion-2-1-base, stabilityai/stable-diffusion-xl-base-1.0, stabilityai/stable-diffusion-2, stabilityai/stable-diffusion-2-1, stabilityai/stable-diffusion-2-base, CompVis/stable-diffusion-v1-2, CompVis/stable-diffusion-v1-3, stabilityai/stable-diffusion-3.5-medium, stabilityai/stable-diffusion-3-medium-diffusers
segmentation_args=$(cat << EOS
save_cross_att=${save_cross_att}
use_cls_predict=${use_cls_predict}
elbo_timesteps=[[1,999,50]]
collect_timesteps=[[20,201,20]]
diffusion.variant=stabilityai/sdxl-turbo
cross_gaussian_var=3
elbo_strength=1
EOS
)

# evaluation.py arguments using here document, see full list of options in ./config/method/evaluation.yaml
evaluation_args=$(cat << EOS
save_mask=${save_mask}
EOS
)

for dataset in ${datasets[@]}
do
    # 1. optional classification
    if [ "${use_cls_predict}" = "True" ]; then
        CUDA_VISIBLE_DEVICES=${device} python classification.py --dataset-cfg ./configs/dataset/${dataset}.yaml output_path.${dataset}=${output_path}/${dataset}${dataset_suffix} data_name_list=./data/${dataset}/val_id${dataset_suffix}.txt ${classification_args}
    fi
    # 2. segmentation
    CUDA_VISIBLE_DEVICES=${device} python segmentation_with_elbo.py --dataset-cfg ./configs/dataset/${dataset}.yaml output_path.${dataset}=${output_path}/${dataset}${dataset_suffix} data_name_list=./data/${dataset}/val_id${dataset_suffix}.txt ${segmentation_args}
    # 3. evaluation
    python evaluation.py --dataset-cfg ./configs/dataset/${dataset}.yaml output_path.${dataset}=${output_path}/${dataset}${dataset_suffix} data_name_list=./data/${dataset}/val_id${dataset_suffix}.txt ${evaluation_args}
done