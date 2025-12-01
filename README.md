<div align="center">
<h1>CADFA-Net</h1>
<h3>Collaborative Attention and Dual-Domain Feature Aggregation Network
for Underwater Dam Crack Segmentationn</h3>
</div>

## UDCD Dataset
### The complete dataset can be obtained by contacting the corresponding author via email(xinyx@hhu.edu.cn).

Underwater crack images collected from real dam inspection locations. We equipped professional underwater cameras on a robotic system to capture video footage of underwater dam structures. We extracted frames from these videos and filtered out images of intact dams, resulting in 250 valid underwater crack images. To expand the dataset and enhance model generalization, we applied data augmentation techniques such as rotation, cropping, and mirroring. This resulted in a final dataset of 1,200 images, each with a resolution of 448 × 448 pixels. This dataset provides essential support for underwater crack detection research.

The UDCD dataset presents significant complexity and challenges, as evidenced by the following aspects: the images cover various typical types of dam cracks (such as longitudinal cracks, transverse cracks, mesh cracks, etc.). The acquisition environment involves varying water transparency, light intensity and direction, observation angles, among others. The dataset construction fully considers actual interferences in the underwater environment, such as turbid water, insufficient or uneven lighting, strong reflection interference, obstruction by floating debris, and camera shake. These factors result in images with an overall yellowish hue, reduced contrast, difficulties in distinguishing cracks from the background in terms of brightness and color characteristics, blurred edges, and complex background textures, significantly increasing the difficulty of the image segmentation task and highlighting the high representativeness and research value of this dataset for practical applications.

To provide a more intuitive understanding of the UDCD dataset, we present several representative examples. These samples illustrate the diverse crack types, complex backgrounds, and challenging visual conditions present in the dataset, including issues such as low contrast, turbidity, and uneven illumination. 

### Example Images

<img src="Sample Images.png" alt="Dataset Samples">


</details>

## Getting Started

### Installation

#### 1. Clone the LocalMamba repository:

```shell
https://github.com/shblyy/CADFA-Net.git
```

#### 2. Environment setup:
```shell
conda create -n CADFA python=3.10
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

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
