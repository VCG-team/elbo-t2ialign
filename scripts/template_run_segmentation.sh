#!/bin/bash
# find least used gpu, min memory first, min utilization second
# set device to "" to use all gpus, or set device to specific gpu indexes to use them, e.g. "0,1,2"
device=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | sed '1d' | sort -t ',' -k 2,2n -k 3,3n | head -n 1 | cut -d ',' -f 1)

# get script name as output folder name(remove file extension)
output_folder=$(basename $0 | rev | cut -d '.' -f 2- | rev)
output_path="./output/${output_folder}"

# experiment datasets
datasets=("voc" "context" "coco" "ade20k" "aep")
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

# run_segmentation.py arguments using here document, see full list of options in ./configs/run_segmentation.yaml
# diffusion.variant options: stable-diffusion-v1-5/stable-diffusion-v1-5, CompVis/stable-diffusion-v1-4, stabilityai/sdxl-turbo, stabilityai/sd-turbo, stabilityai/stable-diffusion-2-1-base, stabilityai/stable-diffusion-xl-base-1.0, stabilityai/stable-diffusion-2, stabilityai/stable-diffusion-2-1, stabilityai/stable-diffusion-2-base, CompVis/stable-diffusion-v1-2, CompVis/stable-diffusion-v1-3, stabilityai/stable-diffusion-3.5-medium, stabilityai/stable-diffusion-3-medium-diffusers, playgroundai/playground-v2.5-1024px-aesthetic
segmentation_args=$(cat << EOS
save_cross_att=${save_cross_att}
elbo_timesteps=[[1,999,50]]
collect_timesteps=[[20,201,20]]
diffusion.variant=stable-diffusion-v1-5/stable-diffusion-v1-5
cross_gaussian_var=3
elbo_strength=3
EOS
)

# evaluate_segmentation.py arguments using here document, see full list of options in ./configs/evaluate_segmentation.yaml
evaluation_args=$(cat << EOS
save_mask=${save_mask}
EOS
)

for dataset in ${datasets[@]}
do
    # 1. segmentation
    if [ "${dataset}" = "aep" ] || [ "${dataset}" = "png" ]; then
        segmentation_args="${segmentation_args} elbo_text.type=file source_text.type=file"
    fi
    CUDA_VISIBLE_DEVICES=${device} python run_segmentation.py --dataset-cfg ./metadata/${dataset}/info.yaml output_path=${output_path}/${dataset}${dataset_suffix} data_name_list=./metadata/${dataset}/val_id${dataset_suffix}.txt ${segmentation_args}
    # 2. evaluation
    python evaluate_segmentation.py --dataset-cfg ./metadata/${dataset}/info.yaml output_path=${output_path}/${dataset}${dataset_suffix} data_name_list=./metadata/${dataset}/val_id${dataset_suffix}.txt ${evaluation_args}
done