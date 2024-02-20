### Verify the system has a CUDA-capable GPU.
 ``` lspci | grep -i nvidia ```
01:00.0 VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP104 High Definition Audio Controller (rev a1)

### Verify the system is running a supported version of Linux.
```uname -m && cat /etc/*release```

x86_64
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"
PRETTY_NAME="Ubuntu 22.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy

### Verify the system has gcc installed.
```  gcc --version ```
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

### Verify the system has the correct kernel headers and development packages installed.
```uname -r```
6.5.0-17-generic
### Download the NVIDIA CUDA Toolkit.
```sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --dkms```
## Handle conflicting installation methods.

nvidia-smi
nvcc --version

12.02.24

sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

sudo reboot

nvidia-smi # not what I expected
nvcc --version # not what I expected

sudo apt install nvidia-cuda-toolkit


sudo reboot

nvidia-smi # not what I expected
nvcc --version

sudo apt install nvidia-utils-535 

sudo reboot

nvidia-smi # not what I expected
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

sudo apt install nvidia-driver-535


conda remove pytorch torchvision torchaudio cudatoolkit=11.




## new try for solving NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

sudo apt update
apt list --upgradable

sudo apt install --reinstall nvidia-driver
 = >E: Unable to locate package nvidia-driver

sudo apt install --reinstall xserver-xorg-video-nvidia-535


sudo apt install --reinstall nvidia-driver
 = >E: Unable to locate package nvidia-driver

