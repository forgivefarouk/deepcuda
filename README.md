# DeepCUDA

DeepCUDA is a repository dedicated to writing and optimizing CUDA kernels for high-performance computing, deep learning, and general-purpose GPU acceleration. The goal is to explore efficient parallel computation techniques, optimize memory usage, and push the limits of GPU programming.

## Features

- **High-Performance Computing:** Leverage CUDA for advanced GPU computations.
- **Deep Learning:** Optimize kernels for deep learning applications.
- **General-Purpose GPU Acceleration:** Explore and implement various GPU-based acceleration techniques.

## Setup: Running CUDA in Google Colab

The following steps demonstrate how to remove existing CUDA installations, install the latest CUDA toolkit on Ubuntu 24.04, and load the `nvcc` plugin for running CUDA code directly in a Colab environment.

### 1. Remove Existing CUDA Installations

Run the following commands to purge any existing CUDA and NVIDIA installations:

```bash
!apt-get --purge remove -y cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove -y cuda-*
!apt autoremove -y
!apt-get update
```
### 2. Install the Latest CUDA for Ubuntu 24.04

```bash
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
!sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
!sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
!sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update
!sudo apt-get -y install cuda-toolkit-12-8
```
### 3. Load the NVCC Plugin
```bash
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
### 4. Example: Running a CUDA Program
```bash
!nvcc -arch=sm_75 -gencode=arch=compute_75,code=sm_75 vector_add.cu -o vector_add
!./vector_add
```

