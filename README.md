# Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification (TNNLS 2023)

### Di Wang, Bo Du, and Liangpei Zhang

### Pytorch implementation of our [paper](https://arxiv.org/pdf/2106.13952.pdf) for graph convolution based hyperspectral image classification.

<figure>
<img src=model.png>
<figcaption align = "center"><b>Figure - The proposed SSGRN. </b></figcaption>
</figure>


### How to create env


```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
#?? do i need this
~/miniconda3/bin/conda init bash 


```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
```

```
git clone https://github.com/HSI-Analysis/MSSGRN.git

conda create --name MSSGRN python=3.8

conda activate MSSGRN

pip install -r requirements.txt
```

### How to run
```
cd Desktop/HSI/Models/MSSGRN/
conda activate MSSGRN

cd utils/src
python setup.py install
or
python -m pip install .

cd ../../

python trainval.py

```

##Notes
scp -r Dataset user181010.2.0.13:~/Desktop/HSI/Dataset
scp -r HSI-Data user181010.2.0.13:~/Desktop/HSI
scp -r ~/Desktop/HSI/HSI-Data/data_size=696x520/porcine2_696x520x31 user181010.2.0.13:~/Desktop/HSI/HSI-Data/data_size=696x520
scp -r trainval.py user1810@10.2.0.13:~/Desktop/HSI/Models/MSSGRN
scp -r test.py user1810@10.2.0.13:~/Desktop/HSI/Models/MSSGRN
scp -r setup.py user1810@10.2.0.13:~/Desktop/HSI/Models/MSSGRN/utilts/src
ssh user1810@10.2.0.13
1810

 scp user1810@10.2.0.13:~/Desktop/HSI/Models/MSSGRN/segrn_afeyan_experiment_0_valbest_tmp.pth /Users/apple/Downloads
segrn_afeyan_experiment_0_valbest_tmp.pth

06.02.24
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102



### Installing PyTorch with GPU 

##### Step 1: Install NVIDIA GPU Drivers
```
sudo apt install nvidia-driver-535
sudo reboot
```
/home/user1810/miniconda3/bin/nvcc

##### Step 2: Install cuDNN
```
conda install -c nvidia cuda-nvcc
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc
```

#### Step 3: Check CUDA version:

```
nvcc --version
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0


#### Step 4: Install CUDA Toolkit:

```
conda install -c anaconda cudatoolkit
```

#### Step 5: Install PyTorch:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f 
https://download.pytorch.org/whl/torch_stable.html


pip install torch==1.9.0+cu12.3 torchvision==0.10.0+cu12.3 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html



conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.9.0+cu123 torchvision==0.10.0+cu123 torchaudio==0.9.0 -f 
https://download.pytorch.org/whl/torch_stable.html
```

###
git clone https://github.com/NVIDIA/apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

07.02.24
conda install -c nvidia cudatoolkit=11.1

conda install -c nvidia/label/cuda-11.1 cuda-nvcc

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html




11.02.2024
### Verify the system has a CUDA-capable GPU.
lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP104 High Definition Audio Controller (rev a1)

### Verify the system is running a supported version of Linux.
 lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy

### Verify the system has gcc installed.
gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

### Verify the system has the correct kernel headers and development packages installed.
uname -r
6.5.0-17-generic


dpkg -l | grep linux-headers-$(uname -r)
ii  linux-headers-6.5.0-17-generic           6.5.0-17.17~22.04.1                     amd64        Linux kernel headers for version 6.5.0 on 64 bit x86 SMP


 dpkg -l | grep build-essential
ii  build-essential                          12.9ubuntu3                             amd64        Informational list of build-essential packages
## Usage
Step 1: Install NVIDIA GPU Drivers

1. Install Pytorch 1.9 with Python 3.8.
2. Clone this repo.
```
git clone https://github.com/DotWang/SSGRN.git
```
3. Prepare a suitable GCC version, then install the SSN
```
cd utils/src
python setup.py install
```
4. Training, validation, testing and prediction with ***trainval.py*** :

For example, when implementing SSGRN on [Salinas Valley dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

```
 CUDA_VISIBLE_DEVICES=0 python -u trainval.py \
    --dataset 'salina' --network 'ssgrn' \
    --norm 'norm' \
    --input_mode 'whole' \
    --input_size 128 128 --overlap_size 54 \
    --experiment-num 10 --lr 1e-3 \
    --epochs 1000 --batch-size 1 \
    --val-batch-size 1 \
    --se_groups 256 --sa_groups 256
```
Then the evaluated accuracies, the trained models and the classification maps are separately saved.

When training on the Houston dataset, using the mode of `part` and setting the input_size

```
    --input_mode 'part' \
    --input_size 349 635 --overlap_size 0 \
```

## Paper and Citation

If this repo is useful for your research, please cite our [paper](https://arxiv.org/abs/2106.13952).

```
@ARTICLE{wang_2023_ssgrn,
  author={Wang, Di and Du, Bo and Zhang, Liangpei},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2023.3265560}}

```

## Thanks
[GCN-Pytorch](https://github.com/tkipf/pygcn) &ensp; [MDGCN](https://github.com/LEAP-WS/MDGCN) &ensp; [SSN](https://github.com/NVlabs/ssn_superpixels) &ensp; [SSN-Pytorch](https://github.com/perrying/ssn-pytorch)


## Relevant Projects
[1] <strong> Pixel and Patch-level Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Adaptive Spectral–Spatial Multiscale Contextual Feature Extraction for Hyperspectral Image Classification, IEEE TGRS, 2020 | [Paper](https://ieeexplore.ieee.org/document/9121743/) | [Github](https://github.com/DotWang/ASSMN)
<br> <em> &ensp; &ensp;  Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang and Yonghao Xu</em>

[2] <strong> Image-level/Patch-free Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; Fully Contextual Network for Hyperspectral Scene Parsing, IEEE TGRS, 2021 | [Paper](https://ieeexplore.ieee.org/document/9347487) | [Github](https://github.com/DotWang/FullyContNet)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, and Liangpei Zhang</em>
 
[3] <strong> Neural Architecture Search for Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; HKNAS: Classification of Hyperspectral Imagery Based on Hyper Kernel Neural Architecture Search, IEEE TNNLS, 2023 | [Paper](https://ieeexplore.ieee.org/document/10159237) | [Github](https://github.com/DotWang/HKNAS)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Bo Du, Liangpei Zhang, and Dacheng Tao</em>

[4] <strong> ImageNet Pretraining and Transformer based Hyperspectral Image Classification </strong> 
<br> &ensp; &ensp; DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification, IEEE TIP, 2023 | [Paper](https://ieeexplore.ieee.org/document/10112639) | [Github](https://github.com/DotWang/DCN-T)
 <br><em> &ensp; &ensp; Di Wang<sup>&#8727;</sup>, Jing Zhang, Bo Du, Liangpei Zhang, and Dacheng Tao</em>
