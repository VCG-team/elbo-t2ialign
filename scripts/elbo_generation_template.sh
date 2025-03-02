#!/bin/bash
# find least used gpu, min memory first, min utilization second
# set device to "" to use all gpus, or set device to specific gpu indexes to use them, e.g. "0,1,2"
device=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | sed '1d' | sort -t ',' -k 2,2n -k 3,3n | head -n 1 | cut -d ',' -f 1)

# get script name as output folder name(remove file extension)
output_folder=$(basename $0 | rev | cut -d '.' -f 2- | rev)
output_path="./output/${output_folder}"

# experiment datasets
datasets=("ane" "dvmp" "abc6k")

# generation_with_elbo.py arguments using here document, see full list of options in ./config/method/generation.yaml
# diffusion.variant options: stable-diffusion-v1-5/stable-diffusion-v1-5, CompVis/stable-diffusion-v1-4, stabilityai/sdxl-turbo, stabilityai/sd-turbo, stabilityai/stable-diffusion-2-1-base, stabilityai/stable-diffusion-xl-base-1.0, stabilityai/stable-diffusion-2, stabilityai/stable-diffusion-2-1, stabilityai/stable-diffusion-2-base, CompVis/stable-diffusion-v1-2, CompVis/stable-diffusion-v1-3
generation_args=$(cat << EOS
diffusion.variant=stable-diffusion-v1-5/stable-diffusion-v1-5
elbo_strength=1.2
EOS
)

# evaluate_generation.py arguments using here document, see full list of options in ./config/method/evaluation.yaml
# clip.variant options: openai/clip-vit-large-patch14, openai/clip-vit-base-patch32
evaluation_args=$(cat << EOS
clip.variant=openai/clip-vit-base-patch32
EOS
)

for dataset in ${datasets[@]}
do
    # 1. generation
    CUDA_VISIBLE_DEVICES=${device} python generation_with_elbo.py --dataset-cfg ./configs/dataset/${dataset}.yaml output_path.${dataset}=${output_path}/${dataset} ${generation_args}
    # 2. evaluation
    python evaluate_generation.py --dataset-cfg ./configs/dataset/${dataset}.yaml output_path.${dataset}=${output_path}/${dataset} ${evaluation_args}
done