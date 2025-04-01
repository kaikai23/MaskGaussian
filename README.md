# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks [CVPR 2025]

<div id="top" align="center">
 
<a href="https://arxiv.org/abs/2412.20522"><img src="https://img.shields.io/badge/Arxiv-2412.20522-B31B1B.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Project-Page-048C3D"></a>
<a href="https://github.com/kaikai23/MaskGaussian"><img src="https://img.shields.io/github/stars/kaikai23/MaskGaussian"></a>
</div>


## :mega: Updates
[03/2025] 🎈: Post-training code is released. Now you can also directly use MaskGaussian to prune an already trained 3D-GS!

[02/2025] Accepted to [CVPR 2025](https://cvpr.thecvf.com/).

[01/2025] We release the code.

## Overview
We introduce MaskGaussian to prune Gaussians while retaining reconstruction quality. It dynamically samples a subset of Gaussians to render the scene during training. Not sampled Gaussians also receive gradients through [mask-diff-gaussian-rasterization](https://github.com/kaikai23/mask-diff-gaussian-rasterization) and update their chance to be used in future iterations.

Our method improves rendering speed, reduces model size, GPU memory, and training time, and supports both training from scratch and post-training refinement.

<img height="250" alt="image" src="https://github.com/user-attachments/assets/4855522d-9fb2-4044-90f2-1ff9cb62b1d1" />


## Installation
1. **Clone the repository**
```
git clone --recursive https://github.com/kaikai23/MaskGaussian.git
cd MaskGaussian
```
2. **Install dependencies**
```
conda create -n maskgs python=3.9
conda activate maskgs
pip install "numpy<2.0" plyfile tqdm icecream torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit  # can be skipped if cuda-11.8 is already installed
CUDA_HOME=PATH/TO/CONDA/envs/maskgs pip install submodules/mask-diff-gaussian-rasterization submodules/diff-gaussian-rasterization submodules/simple-knn/
```

## Data Preparation
First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── gs_datasets
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

### Public Data

- The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). 
- The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Training and Evaluation in One Go
To train, render and evaluate our method on the 3 datasets in the paper, simply run:
```
python run_all.py
```
The training output is logged in `train.log` under the `output/scene_name` folder, and the final metrics are recorded in `results.json` under the same folder. The training time can be read from `train.log`, and GPU memory consumption can be read from `GPU_mem` card in tensorboard records by running `tensorboard --logdir /path/to/output`.
Finally, note that **the output of our method is 100% in vanilla format and can be viewed directly in any 3dgs viewer**, such as popular [SuperSplat](https://superspl.at/editor) and [antimatter15](https://antimatter15.com/splat/).

## Training a single scene
To train a single scene, run:
```
python train.py -s /path/to/input_scene --eval -m /path/to/output
```
with optional parameters:

• **--lambda_mask**: the coefficient of mask loss

• **--mask_from_iter**: the start iteration for mask loss

• **--mask_until_iter**: the end iteration for mask loss

There are 3 settings in the paper, and their configurations can be found in `run_all.py`.

Last, to render and evaluate the test set, run:
```
python render.py -m /path/to/output --skip_train
python metrics.py -m /path/to/output
```
Since we do not save mask, no special handling is required for evaluation.

## Post-training and evaluation
To prune an already trained 3DGS, specify its checkpoint path in `scripts/run_prune_finetune.sh` and run:
```
bash scripts/run_prune_finetune.sh
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## TODO List
- \[ \] Code of MaskGaussian + Taming-3DGS.
- \[x\] Support post-training.

## Contact

- Yifei Liu: liuyifei@pjlab.org.cn

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{liu2024maskgaussianadaptive3dgaussian,
      title={MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks}, 
      author={Yifei Liu and Zhihang Zhong and Yifan Zhan and Sheng Xu and Xiao Sun},
      year={2024},
      eprint={2412.20522},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.20522}, 
}</code></pre>
  </div>
</section>

## Acknowledgement

This project is built upon [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Compact-3DGS](https://github.com/maincold2/Compact-3DGS), and [LightGaussian](https://github.com/VITA-Group/LightGaussian). We thank all authors for their great work!
