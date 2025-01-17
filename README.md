# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks

### [[Paper(arxiv)](https://arxiv.org/abs/2412.20522)]

## Method Overview
**MaskGaussian** is a method for pruning the number of Gaussians while preserving high reconstruction quality. It treats Gaussian points as probabilistically existing, and samples them to render a scene. Unsampled Gaussians do not affect the rendering outcome but can still receive meaningful gradients to dynamically adjust their usage probability. This is enabled by our [mask-diff-gaussian-rasterization](https://github.com/kaikai23/mask-diff-gaussian-rasterization), which applies masks during the rasterization process instead of multiplying them with Gaussian attributes.

## Installation
1. **Clone the repository**
```
git clone --recursive https://github.com/kaikai23/MaskGaussian.git
cd MaskGaussian
```
2. **Create conda environment**
```
conda create -n maskgs python=3.9
conda activate maskgs
```
3. **Install Dependencies**
```
pip install "numpy<2.0" plyfile tqdm torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
4. **Install Submodules**
```
CUDA_HOME=PATH/TO/CONDA/envs/maskgs pip install submodules/mask-diff-gaussian-rasterization submodules/simple-knn/
```

## Data Preparation
Please follow the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to prepare the datasets: Mip-Nerf360, Tanks&Temples, and Deep Blending.

## Running
1. **Train the model**
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval -m <output_folder>
```
2. **Save the result in original 3DGS format**
```
python save_ply_nomask.py -m <output_folder>
```
The result will be save to `<outputfolder>/point_cloud/<last_iteration+1>/point_cloud.ply`, and can be rendered and viewed just like original 3DGS.

## Evaluation
We recommend directly using the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to evaluate the resulting ply file from the last step.

## Hyperparameters
The default configuration applies masks throughout the training process and strikes a balance between reconstruction quality, number of primitives and GPU consumption. 

To use masks in different phases, the following hyperparameters should be changed:

 - **--lambda_mask**: the coefficient of the mask loss added to the total loss. Default: 0.0005
 - **--mask_type**: have 3 options. 'constant': use masks all the way. 'halfway': use masks after 15,000 iterations. 'late': use masks during 19,000~20,000 iterations. Default: 'constant'.

We provide 2 recommended configurations:<br/>
**Best PSNR + least number of Gaussian primitives**: `--lambda_mask 0.1 --mask_type late` <br/>
**Save GPU memory in training**: `--lambda_mask 0.001 --mask_type constant`


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
