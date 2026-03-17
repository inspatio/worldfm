# WorldFM

<br/>

<div align="center">
    <img src="resources/worldfm.jpg"/>
</div>
<br/>

<div align="left">
<div align="left">

[![LICENSE](https://img.shields.io/github/license/inspatio/worldfm)](https://github.com/inspatio/worldfm/blob/main/LICENSE)  [![arXiv](https://img.shields.io/badge/arXiv-2603.11911-b31b1b.svg)](https://arxiv.org/abs/2603.11911)


WorldFM, a real-time multi-view diffusion model. Given a reference image and target camera poses, WorldFM generates images at those new viewpoints. Checkout our website ([WorldFM](https://inspatio.github.io/worldfm)) for videos and interactive results!

## Installation

### 1. Create Conda Environment

```bash
# Edit CONDA_ENV_PATH in setup.sh to your desired prefix first
bash setup.sh
```

This will:

- Create the `WorldFM` conda environment (Python 3.10, PyTorch 2.5, CUDA 12.4)
- Install pip dependencies from `requirements.txt`
- Initialize git submodules (HunyuanWorld-1.0, MoGe, Real-ESRGAN, ZIM)
- Build Real-ESRGAN and ZIM in development mode

### 2. Manual Setup (alternative)

```bash
conda env create -f WorldFM.yaml --prefix /path/to/envs/WorldFM
conda activate /path/to/envs/WorldFM
pip install -r requirements.txt
git submodule update --init --recursive
cd submodules/MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925

# HunyuanWorld-1.0 requirements
cd ../Real-ESRGAN
pip install basicsr-fixed facexlib gfpgan
python setup.py develop
cd ../ZIM
pip install -e .
```

For consistent scene generation, we employ an internal generative model that is not included in the open-source release.
To support reproducibility, users can integrate alternative open-source panorama generation models (e.g., HunyuanWorld-1.0). This substitution does not impact the core spatial reasoning framework of WorldFM.

## Getting Started

### Download Pretrained Model

Download model checkpoints from [huggingface](https://huggingface.co/inspatio/worldfm) by running:

```sh
python download_ckpts.py
```

You will get:

```
weights/
  ├── vae/
  ├── worldfm_1-step.pth  # DMD step=1, faster
  └── worldfm_2-step.pth  # DMD step=2, better quality
```

Use `--step 1` or `--step 2` in `run_pipeline.py` to select the corresponding model.

## Usage

### Demo

We provide a sample scene with a pre-defined camera trajectory in `demo/`. Run the following command to generate an MP4 video along the trajectory:

```bash
python run_pipeline.py --meta demo/meta.json --output_dir outputs
```

The output video will be saved to `outputs/<scene_name>/output.mp4`.

### Input Format

Prepare a `meta.json` file:

Single pose:

```json
{
  "name": "scene_001",
  "image": "input.jpg",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "c2w": [
    [r00, r01, r02, tx],
    [r10, r11, r12, ty],
    [r20, r21, r22, tz],
    [  0,   0,   0,  1]
  ]
}
```

Multiple poses (generates one output per pose):

```json
{
  "name": "scene_001",
  "image": "input.jpg",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "c2w": [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[...], [...], [...], [...]],
    ...
  ]
}
```

- **name**: scene identifier, used as the output subdirectory name
- **image**: relative path (from `meta.json` location) to the input perspective image
- **K**: 3×3 camera intrinsic matrix
- **c2w**: a single 4×4 or a list of N×4×4 camera-to-world matrices (target viewpoints)

### Run Inference with Your Own Data

```bash
# Default: output as MP4 video
python run_pipeline.py --meta <META_JSON> --output_dir <OUTPUT_DIR>

# Save per-frame PNG images instead
python run_pipeline.py --meta <META_JSON> --output_dir <OUTPUT_DIR> --save_mode image
```

### Configuration

Default parameters are defined in `default.yaml`. Override them via:

1. **CLI arguments** (highest priority)
2. **Custom config file**: `--config my_config.yaml`
3. `**default.yaml`** (lowest priority)

### Output

With `--save_mode video` (default):

```
<output_dir>/<name>/
  └── output.mp4          # Video composed of all generated frames
```

With `--save_mode image`:

```
<output_dir>/<name>/
  ├── output.png           # Single pose
  # or
  ├── output_0000.png      # Multiple poses
  ├── output_0001.png
  └── ...
```

# License

The license of our codebase is [Apache-2.0](https://github.com/inspatio/worldfm/blob/main/LICENSE). Note that this license only applies to code in our library, the dependencies and submodules of which ([HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0/blob/main/LICENSE), [MoGe](https://github.com/microsoft/MoGe/blob/main/LICENSE)) are separate and individually licensed.

# Contributing

We appreciate all contributions to improve WorldFM.

# Citing

If you use WorldFM in your research, please use the following BibTeX entry.

```bib
@misc{worldfm,
    title={Inspatio-WorldFM: An Open-Source Real-Time Generative Frame Model for Spatial Intelligence},
    author={WorldFM Contributors},
    howpublished = {\url{https://github.com/inspatio/worldfm}},
    year={2026}
}
```

# Acknowledgement

This codebase is built upon [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma). We would like to express our gratitude to the PixArt Team for open-sourcing their code and models. Their contributions have been instrumental to the development of this project. We also appreciate [PRoPe](https://github.com/liruilong940607/prope), [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git) and [MoGe](https://github.com/microsoft/MoGe.git) for their excellent work.