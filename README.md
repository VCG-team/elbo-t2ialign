<h1 align='center'>ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models</h1>
<p align="center"> <span style="color:#137cf3; font-family: Gill Sans">Qin Zhou</span>, <span style="color:#137cf3; font-family: Gill Sans">Zhiyang Zhang</span>, <span style="color:#137cf3; font-family: Gill Sans">Jinglong Wang</span>, <span style="color:#137cf3; font-family: Gill Sans">Xiaobin Li</span>, <span style="color:#137cf3; font-family: Gill Sans">Jing Zhang</span><sup>*</sup>, <span style="color:#137cf3; font-family: Gill Sans">Qian Yu</span>, <span style="color:#137cf3; font-family: Gill Sans">Lu Sheng</span>, <span style="color:#137cf3; font-family: Gill Sans">Dong Xu</span> <br> 
<span style="font-size: 16px">Beihang University</span>, <span style="font-size: 16px">University of Hong Kong</span></p>

<div align="center">
  <a href="https://vcg-team.github.io/elbo-t2ialign-webpage/"><img src="https://img.shields.io/static/v1?label=elbo-t2ialign&message=Project&color=purple"></a>  
  <a href="https://arxiv.org/abs/2506.09740"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a>  
  <a href="https://github.com/VCG-team/elbo-t2ialign"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github"></a>  
  <a href="https://huggingface.co/datasets/Matrix53/elbo-t2ialign"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface"></a>
</div>
<br>
<p align="center">
  <img src="pipeline.png"  height=400>
</p>

## 📺 Abstract

Diffusion models excel at image generation. Recent studies have shown that these models not only generate high-quality images but also encode text-image alignment information through attention maps or loss functions. This information is valuable for various downstream tasks, including segmentation, text-guided image editing, and compositional image generation. However, current methods heavily rely on the assumption of perfect text-image alignment in diffusion models, which is not the case. In this paper, we propose using zero-shot referring image segmentation as a proxy task to evaluate the pixel-level image and class-level text alignment of popular diffusion models. We conduct an in-depth analysis of pixel-text misalignment in diffusion models from the perspective of training data bias. We find that misalignment occurs in images with small-sized, occluded, or rare object classes. Therefore, we propose ELBO-T2IAlign—a simple yet effective method to calibrate pixel-text alignment in diffusion models based on the evidence lower bound (ELBO) of likelihood. ELBO-T2IAlign is training-free and generic: it requires no additional annotations, model retraining, or architectural modifications, and it can be directly applied to different diffusion backbones. Extensive experiments on zero-shot referring image segmentation, text-guided image editing, and compositional image generation verify that the proposed calibration improves pixel-text alignment across complementary downstream tasks.

## 🛠️ Setup
```bash
# clone repository and install git lfs
git clone https://github.com/VCG-team/elbo-t2ialign
cd elbo-t2ialign
git lfs install
git lfs pull

# create conda env with environment.yaml
conda env create -f environment.yaml
conda activate elbo-t2ialign
python -m spacy download en_core_web_sm

# download datasets from Hugging Face
pip install 'huggingface_hub[cli]'
hf download Matrix53/elbo-t2ialign --repo-type=dataset --local-dir ./datasets

# unzip dataset and clean zipped files
cd datasets
cat dataset.tar.gz.* > dataset.tar.gz
tar -xzf dataset.tar.gz
mv datasets_copy/* ./
rm -r dataset*
cd ..
```

## 🎨 Usage

To run segmentation experiment, you can use `./scripts/template_run_segmentation.sh`:
```bash
# You can change arguments in the script below to test different settings(include models/hyperparameters etc.)
# Full arguments are in `configs/run_segmentation.yaml`
./scripts/template_run_segmentation.sh
```

To run composable generation experiment, you can use `./scripts/template_run_generation.sh`:
```bash
# You can change arguments in the script below to test different settings(include models/hyperparameters etc.)
# Full arguments are in `configs/run_generation.yaml`
./scripts/template_run_generation.sh
```

If you just want to visualize a few results, you can use `scripts/run_segmentation.ipynb`(segmentation) and `scripts/run_generation.ipynb`(generation).

Welcome to open an issue if you have any question.

## 🥳 Credits

We appreciate all open source projects that we use in this project:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers)
- [MCTFormer](https://github.com/xulianuwa/MCTformer), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [clip-es](https://github.com/linyq2117/CLIP-ES)

## 🔗 Prior Work

ELBO-T2IAlign is a follow-up to our previous paper, **[DiffSegmenter](https://github.com/VCG-team/DiffSegmenter)**. The prior work explores how diffusion models encode text-image alignment information for downstream tasks, while ELBO-T2IAlign further examines the limitations of such alignment and calibrates pixel-level text-image correspondence through an ELBO-based objective.

For more details, please visit the previous project page: [DiffSegmenter](https://github.com/VCG-team/DiffSegmenter).

## 💕 Citation
```bibtex
@article{zhou2025elbo,
    title={ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models},
    author={Zhou, Qin and Zhang, Zhiyang and Wang, Jinglong and Li, Xiaobin and Zhang, Jing and Yu, Qian and Sheng, Lu and Xu, Dong},
    journal={arXiv preprint arXiv:2506.09740},
    year={2025}
}
```

## ⭐️ Star History

![Star History Chart](https://api.star-history.com/svg?repos=VCG-team/elbo-t2ialign&type=Date)
