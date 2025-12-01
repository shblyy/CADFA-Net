<div align="center">
<h1>CADFA-Net</h1>
<h3>Collaborative Attention and Dual-Domain Feature Aggregation Network
for Underwater Dam Crack Segmentationn</h3>
</div>


</details>

## Getting Started

### Installation

#### 1. Clone the LocalMamba repository:

```shell
https://github.com/shblyy/CADFA-Net.git
```

#### 2. Environment setup:
conda create -n CADFA python=3.10
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

_**Install Mamba kernels:**_
```shell
cd causual-conv1d && pip install .
cd ..
cd mamba-1p1p1 && pip install .
```


Other dependencies:
```shell
timm==0.9.12
fvcore==0.1.5.post20221221
```


## Image Classification

### Dataset

We use ImageNet-1K dataset for training and validation. It is recommended to put the dataset files into `./data` folder, then the directory structures should be like:
```
classification
├── lib
├── tools
├── configs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
│   │   ├── cifar-100-python
```

## Acknowledgment

This project is based on VMamba, Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), [OpenMMLab](https://github.com/open-mmlab),
and the `analyze/get_erf.py` is adopted from [replknet](https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf), thanks for their excellent works.
