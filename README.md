# PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13+-orange.svg)](https://pytorch.org/)

**PiLoT** is a state-of-the-art pixel-level localization and tracking system for visual localization in large-scale environments. This repository contains the official implementation of our CVPR 2026 paper.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

PiLoT addresses the challenge of real-time visual localization by combining neural rendering with pixel-level feature matching. The system uses a dual-process architecture that simultaneously performs 3D scene rendering and pose estimation, enabling accurate localization in GPS-denied environments.

### Key Contributions

- **Dual-Process Architecture**: Separate rendering and localization processes for real-time performance
- **Pixel-Level Matching**: Direct optimization at the pixel level for improved accuracy
- **Neural Feature Extraction**: Learned feature representations for robust matching
- **ECEF Coordinate System**: Support for large-scale geographic localization

## ✨ Features

- Real-time visual localization with dual-process rendering and tracking
- Support for large-scale 3D tile-based scene rendering
- Pixel-level feature matching with learned descriptors
- Configurable pose refinement with multiple optimization strategies
- Evaluation metrics for position and orientation accuracy

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.7+ (for GPU acceleration)
- OpenSceneGraph (for 3D rendering)
- OsgEarth (for geographic rendering)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/PiLoT.git
cd PiLoT
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate pilot
```

### Step 3: Install Additional Dependencies

```bash
pip install -e .
```

### Step 4: Build 3D Tiles Renderer (Optional)

If you need to use the custom 3D tiles renderer:

```bash
cd 3DTilesRender
mkdir build && cd build
cmake .. -DBUILD_AS_PYTHON_MODULE=ON
make -j$(nproc)
cd ../..
```

### Step 5: Build CUDA Extensions

```bash
cd DirectAbsoluteCostCuda
python setup.py build_ext --inplace
pip install .
cd ..
```

## 📦 Dataset Preparation

### Dataset Structure

Organize your dataset in the following structure:

```
dataset_root/
├── images/
│   └── sequence_name/
│       ├── 0000.png
│       ├── 0001.png
│       └── ...
├── poses/
│   └── sequence_name.txt
├── bbox/
│   └── sequence_name/
│       └── sequence_name_xy.txt
└── target_RTK/
    └── sequence_name_RTK.txt
```

### Pose File Format

The pose file (`sequence_name.txt`) should contain one pose per line:

```
image_name lon lat alt roll pitch yaw
0000.png 114.2604 22.2078 38.8901 0.0 25.0 314.9993
0001.png 114.2605 22.2079 38.8902 0.1 25.1 315.0000
...
```

### Target Points File Format

The target points file (`sequence_name_xy.txt`) should contain:

```
image_name x y
0000.png 1920.0 1080.0
0001.png 1920.0 1080.0
...
```

### Download Datasets

For evaluation datasets, please refer to the specific dataset documentation. Example datasets include:

- Custom UAV sequences
- Large-scale urban scenes
- Synthetic datasets

## 🎮 Usage

### Basic Usage

Run localization on a sequence:

```bash
./run_feicuiwan.sh
```

### Command Line Arguments

```bash
python main.py \
    --config CONFIG_FILE          # Path to configuration file
    --name DATASET_NAME           # Dataset name override
    --init_euler "[p, r, y]"      # Initial Euler angles (optional)
    --init_trans "[x, y, z]"      # Initial translation (optional)
```

### Example Scripts

We provide example scripts for different scenarios:

```bash
# Run on Feicuiwan dataset
bash run_feicuiwan.sh

# Run on Google dataset
bash run_google.sh

# Run on UAV scenes
bash run_uavscenes.sh
```

### Evaluation

After running localization, evaluation metrics are automatically computed:

- **Position Error**: Translation error in meters (XYZ)
- **Orientation Error**: Rotation error in degrees (Euler angles)

Results are saved in the `outputs/` directory.

## ⚙️ Configuration

Configuration files are located in `configs/`. Each configuration file contains:

### Render Configuration

```yaml
render_config:
  model_path: "http://localhost:8078/Scene/Production_6.json"
  render_camera: [width, height, cx, cy, fx, fy]
  max_size:960/512 # render width
  width: 3840  # query width
  height: 2160 # query height
  params: [2700.0, 2700.0, 1915.7, 1075.1] # query [fx, fy, cx, cy]
  distortion: [0.0046, 0.1294, 0, 0.0012, -0.2037] # uery distortion
  dataset_path: "data_demo/query"
```

### Localization Configuration

```yaml
default_confs:
  from_render_test:
    checkpoint: "path/to/checkpoint.ckpt"
    refinement:
      num_dbs: 3
      multiscale: [4, 1]
      point_selection: "all"
    optimizer:
      num_iters: 3
      lambda_: 0.01
      loss_fn: "scaled_barron(0, 0.1)"
    extractor:
      name: "unet_fusion"
      encoder: "mobileone_s0"
      output_dim: [32, 32, 32]
  
  cam_query:
    max_size: render width  
    width: w
    height: h
    params: [fx, fy, cx, cy]
    distortion: [k1, k2, p1, p2, k3]
  
  dataset_path: "/path/to/dataset"
  dataset_name: "sequence_name"
  num_init_pose: 64  # seeds
  padding: true
```

### Key Parameters

- `num_init_pose`: Number of initial pose candidates
- `num_iters`: Number of optimization iterations
- `multiscale`: Multi-scale feature extraction levels
- `point_selection`: Strategy for 3D point selection


## 🔬 Training (Optional)

<!-- To train your own models: -->

<!-- ```bash
python -m pixloc.pixlib.train experiment_name \
    --conf pixloc/pixlib/configs/train_pixloc_megadepth.yaml
``` -->

See `pixloc/pixlib/README.md` for detailed training instructions.

## 📊 Results

Results are saved in the `outputs/` directory:

- `{dataset_name}.txt`: Estimated poses (lon, lat, alt, roll, pitch, yaw)
- `{dataset_name}_xyz.txt`: Estimated 3D target points
- `{dataset_name}/`: Rendered images with annotations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📝 Citation

If you use PiLoT in your research, please cite:

```bibtex
@inproceedings{cheng2026pilot,
  title={PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization},
  author={Cheng, Xiaoya and Wang, Long and Liu, Yan and Liu, Xinyi and Tan, Hanlin and Liu, Yu and Zhang, Maojun and Yan, Shen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of [PixLoc](https://github.com/cvg/pixloc) as the core registration backbone.
- Uses [DeepAC](https://github.com/WangLongZJU/DeepAC) as the training framework.
- Leverages [OpenSceneGraph](https://www.openscenegraph.org/) for core 3D rendering.
- Data sources and platform supported by [Google Earth](https://earth.google.com/web/) and [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/).

## 📧 Contact

For questions and issues, please open an issue on GitHub or contact the authors.

---

**Note**: This is the official implementation for CVPR 2026. For questions about the paper or code, please refer to the issues section.
