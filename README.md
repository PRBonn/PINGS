<p align="center">

  <h1 align="center">📌 PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map</h1>

  <p align="center">
    <a href="https://github.com/PRBonn/PINGS#run-pings"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/PINGS#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/PINGS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
  <h3 align="center"><a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf">Paper</a> | <a href="https://www.youtube.com/watch?v=XuOLoZdcReA&t=2s">Video</a> | <a href="https://www.youtube.com/watch?v=bbAw_ZC_1fI">Demo</a> </h3>
  <div align="center"></div>
</p>

**TL;DR: PINGS is a LiDAR-visual SLAM system unifying distance fields and radiance fields within a neural point map**

![teaser](https://github.com/user-attachments/assets/0ec8b71d-8902-445c-a07c-f9a7e08fb3e2)

<details>
  <summary>[Demo videos (click to expand)]</summary>


### Rendering from PINGS map of Bonn

| example 1 | example 2 |
| :-: | :-: |
| <video src='https://github.com/user-attachments/assets/bff4c53a-a5b8-4912-86af-7cb1bfa4248f'> | <video src='https://github.com/user-attachments/assets/a8e5be1c-bc64-4f39-92b1-31a6eea48b25'> |

| example 3 | example 4 |
| :-: | :-: |
| <video src='https://github.com/user-attachments/assets/f7f69d77-9b67-4cf0-a491-f57f96f6f940'> | <video src='https://github.com/user-attachments/assets/a6733384-39ff-443a-a26e-66429bf5cf4e'> |


### SLAM demo

| HKU | KITTI | IPB Car |
| :-: | :-: | :-: |
| <video src='https://github.com/user-attachments/assets/d4de3d86-589d-4f84-ad30-102fdac9fb53'> | <video src='https://github.com/user-attachments/assets/332c8633-6ff6-4dc2-bd42-19944cc8cfed'> | <video src='https://github.com/user-attachments/assets/839208c0-18ab-401a-bc67-747889f539b2'> |


</details>

----

## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Robots require high-fidelity reconstructions of their environment for effective operation. Such scene representations should be both, geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, the scalable incremental mapping of both fields consistently and at the same time with high quality remains challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to the state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by leveraging the constraints from the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction.
</details>


## Installation

### Platform requirement

* Ubuntu OS (tested on 20.04)

* With GPU, memory > 8 GB recommended

### 0. Clone the repository

```
git clone git@github.com:PRBonn/PINGS.git --recursive
cd PINGS
```

### 1. Set up conda environment

```
conda create --name pings python=3.10
conda activate pings
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

The commands depend on your CUDA version (check it by `nvcc --version`). You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).


### 3. Install other dependencies including the submodules

```
pip3 install -r requirements.txt
```

## Docker
<details>
  <summary>[Details (click to expand)]</summary>

For the prerequisite, you need to install docker and nvidia container runtime.
Firstly edit your data directory at: `- /your_data_storage_directory_here:/packages/pings/data/` in `docker-compose.yml`.
Then build the image by `make build` and run the container by `make run`. 
Afterwards you can use pings as you wish in the terminal. 

</details>

## Data Preparation

A dataset with both RGB and depth observations (LiDAR or depth camera) with extrinsic and intrinsic calibration parameters is required. Note that the input images are supposed to have been already undistorted.

If only range measurements are available, you can still run PINGS with the `--gs-off` flag while PINGS would degenerate to [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).

To extract individual observations from a ROS bag, you may use the [ROS bag converter tool](https://github.com/YuePanEdward/rosbag_converter).

For your own dataset, you may need to implement a new dataloader class and put it in the `dataset/dataloaders` folder. 
Check [here](https://github.com/PRBonn/PINGS/blob/main/dataset/dataloaders/ipb_car.py) for an example.


## Run PINGS


To check how to run PINGS and which datasets have already been supported, you can use the following command:

```
python3 pings.py -h 
```

To check how to inspect the map built by PINGS and render videos from the map, you can use the following command:

```
python3 inspect_pings.py -h 
```

## Citation

<details>
  <summary>[Details (click to expand)]</summary>


If you use PINGS for any academic work, please cite our original [paper](https://www.roboticsproceedings.org/rss21/p040.pdf).

```
@inproceedings{pan2025rss,
title = {{PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map}},
author = {Y. Pan and X. Zhong and L. Jin and L. Wiesmann and M. Popovi\'c and J. Behley and C. Stachniss},
booktitle= {Robotics: Science and Systems (RSS)},
year = {2025},
codeurl = {https://github.com/PRBonn/PINGS},
url = {https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf}
}
```
</details>

## Contact
If you have any questions, please contact:

- [Yue Pan](https://www.ipb.uni-bonn.de/people/yue-pan/index.html) {[yue.pan@igg.uni-bonn.de]()}


## Acknowledgement

<details>
  <summary>[Details (click to expand)]</summary>

PINGS is built on top of our previous work [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM) and we thank the authors for the following works:

* [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
* [Gaussian Surfels](https://github.com/turandai/gaussian_surfels)
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
* [MonoGS](https://github.com/muskie82/MonoGS)
* [Oxford Spires Dataset](https://github.com/ori-drs/oxford_spires_dataset)

</details>
