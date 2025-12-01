<div align="center">
<h1>CADFA-Net</h1>
<h3>Collaborative Attention and Dual-Domain Feature Aggregation Network
for Underwater Dam Crack Segmentationn</h3>
</div>
### Abstract

The accurate detection of underwater concrete dam cracks is essential for structural health monitoring. Accurately segmenting these cracks from the background is a crucial step in quantifying and assessing the extent of dam structural deterioration. However, underwater images often exhibit low contrast and limited resolution, which contribute to severe issues such as low detection accuracy and high false positive rates in existing methods. To overcome these issues, we propose CADFA-Net, an innovative underwater dam crack segmentation method. The proposed method significantly improves the detection performance of underwater dam cracks through the following innovations. We propose a Collaborative Attention State Space (CASS) block to enable multi-scale feature extraction. It boosts the detection accuracy of fine-grained cracks and maintains the continuity of blurred edge structures. We design a Multimanner Zigzag scanning (MZS) module to extract crack features from multiple orientations. It ensures semantic continuity and enhances the representation of crack topological structures. Finally, a dual-domain feature aggregation module (DDFAM) integrates spatial and frequency domain information. This module effectively separates high-frequency edge details from low-frequency global structures. Adaptive Global Feature Selection (AGFS) is then applied for cross-domain feature fusion, suppressing noise and enhancing crack feature saliency. Experimental results indicate that CADFANet achieves state-of-the-art performance on both underwater and pavement crack datasets, with IoU scores of 75.74% and 70.64%, respectively. The proposed method enhances crack detection accuracy, minimizes false positives from background interference, and demonstrates superior segmentation continuity and robustness.

## Getting Started

### Installation

#### 1. Clone the LocalMamba repository:

```shell
https://github.com/shblyy/CADFA-Net.git
```

#### 2. Environment setup:

We tested our code on `torch==1.13.1` and `torch==2.0.2`.

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
