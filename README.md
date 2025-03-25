# ELBO-T2IAlign

## Setup

1. Clone repository and install git lfs.
```bash
git clone https://github.com/VCG-team/elbo-t2ialign
cd elbo-t2ialign
git lfs install
git lfs pull
```

2. Create [conda](https://conda.io/) env with `environment.yaml`.
```bash
conda env create -f environment.yaml
conda activate elbo-t2ialign
# download spacy model for part-of-speech tags
python -m spacy download en_core_web_sm
```

3. Download datasets from [Google Drive](), and put them in `datasets` folder as follows:
    ```
    elbo-t2ialign
    ├── configs
    ├── utils
    ├── README.md
    ├── ...
    ├── datasets
    │   ├── VOCdevkit
    │   │   ├── VOC2012
    │   │   ├── VOC2010
    │   │   ├── VOCaug
    │   ├── coco_stuff164k
    │   ├── voc_sim
    │   ├── coco_cap
    │   ├── ade
    │   │   ├── ADEChallengeData2016
    │   ├── ABC-6K.txt
    │   ├── AnE.txt
    │   ├── DVMP.txt
    ```

## Usage

Reproducing the results will need about 10G GPU memory.

To test different settings, you can change configuration files in `configs` folder, or pass command line arguments following [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#usage).

We provide experiment script templates `./scripts/template_*` to run experiments. You can modify the script to run different experiments.

Welcome to open an issue if you have any question. 

## Credits

We appreciate all open source projects that we use in this project:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers)
- [MCTFormer](https://github.com/xulianuwa/MCTformer), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [clip-es](https://github.com/linyq2117/CLIP-ES)
- ...

## Citation
```bibtex

```
