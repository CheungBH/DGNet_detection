# DGNet
This repo contains the implementation on detection of the following paper:

**Dynamic Dual Gating Neural Networks**

## Requirements

Our project is developed on [mmdetection](https://github.com/open-mmlab/mmdetection). 

The main requirements of this work are:

- Python 3.7  
- PyTorch == 1.5.0  
- Torchvision == 0.6.0  
- CUDA 10.2

We recommand using conda env to setup the experimental environments.

```shell script
# Create environment
conda create -n DGDet python=3.7 -y
conda activate DGDet

# Install PyTorch & Torchvision
pip install torch==1.5.0 torchvision==0.6.0

# Install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html

# Clone repo
git clone https://github.com/anonymous-9800/DGNet.git ./DGNet
cd ./DGNet/detection

# Create soft link for data
mkdir data
cd data
ln -s ${COCO-Path} ./coco
cd ..

# Install
pip install -r requirements/build.txt
pip install -v -e . 
```

## Running
```shell script
# Train on 8 GPUs
## Retinanet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/retinanet_dg/retinanet_r50_fpn_1x_coco_dg_04.py 8
## Faster_rcnn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/faster_rcnn_dg/faster_rcnn_r50_fpn_1x_coco_dg_04.py 8
```