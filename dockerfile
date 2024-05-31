# Use a base image with Conda installed
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# To use this Dockerfile:
# 1. `nvidia-docker build -t detectron2:v0 .`
# 2. `nvidia-docker run -it --name detectron2 detectron2:v0`
# 
# To enable GUI support (Linux):
# 1. Grant the container temporary access to your x server (will be reverted at reboot of your host): 
#    `xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2``
# 2. `nvidia-docker run -it --name detectron2 --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" detectron2:v0 -v /media/farshid/Data/farshid/dataset/:/CRM_RGBTSeg/datasets/`

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    libpng-dev libjpeg-dev python3-opencv ca-certificates gedit \
    python3-dev build-essential pkg-config git curl wget automake libtool && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision cython \
    'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install opencv-python==4.7.0.72

ENV PATH=/usr/local/cuda/bin:$PATH

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
RUN pip install -e /detectron2_repo

RUN pip install mmcv==1.7.1 pytorch-lightning==1.9.2 scikit-learn==1.2.2 timm==0.6.13 imageio==2.27.0 setuptools==59.5.0

WORKDIR /detectron2_repo

RUN git clone https://github.com/UkcheolShin/CRM_RGBTSeg.git /CRM_RGBTSeg
WORKDIR /CRM_RGBTSeg/models/mask2former/pixel_decoder/ops/

RUN sh make.sh

# Download pretrained weights and convert them
WORKDIR /CRM_RGBTSeg/pretrained
RUN sh download_backbone_pt_weights.sh
RUN sh convert_pth_to_pkl.sh

# Set the container's entrypoint to bash
ENTRYPOINT [ "/bin/bash" ]

ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /CRM_RGBTSeg

# Example run commands:
# RUN wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# RUN python3 demo/demo.py  \
#    --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#    --input input.jpg --output outputs/ \
#    --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
