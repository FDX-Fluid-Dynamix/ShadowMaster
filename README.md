# ShadowMaster

* [General info](#general-info)
* [Python Algorithm](#Python Algorithm)
* [Neuronal Network](#Neuronal Network)

## General info

	
## Python Algorithm

This version to evaluate the Shadowgraphy images is inspired by a Binary thresholding function. The python package
[@OpenCV ](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html) is used and the labeling function from 
[@mahotas](https://mahotas.readthedocs.io/en/latest/labeled.html)



## Neuronal Network

The structure of the network is the [@Mask R-CNN](https://arxiv.org/abs/1703.06870) from 2017 by Kaiming He.
The implementation of the network was done by [@matterport](https://github.com/matterport/Mask_RCNN) ,where a good introduction to the structure of the neural network can be found.

### Setup

All used python packages are listed in the files "Requirements.txt" for cpu and "Requirements_gpu.txt" . For creating the conda Enviroment the following commands are necessary:

conda create -n Env_Name
conda activate Env_Name
conda install --file Requirements.txt
conda install -c conda-forge imgaug
