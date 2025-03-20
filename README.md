# deepcuda
DeepCUDA is a repository dedicated to writing and optimizing CUDA kernels for high-performance computing, deep learning, and general-purpose GPU acceleration. The goal is to explore efficient parallel computation techniques, optimize memory usage, and push the limits of GPU programming.


# Cuda in google colab

## 1.Remove existing CUDA installations
!apt-get --purge remove -y cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove -y cuda-*
!apt autoremove -y
!apt-get update

## 2.Install latest CUDA for Ubuntu 24.04
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
!sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
!sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
!sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update
!sudo apt-get -y install cuda-toolkit-12-8

## 3.load nvcc plugin
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter



# Example to run vector_add.cu file
!nvcc -arch=sm_75 -gencode=arch=compute_75,code=sm_75 vector_add.cu -o vector_add
!./vector_add