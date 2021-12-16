NYCU VRDL Homework1: Nucleus Segementation
# Object Segementation
## Environment
- Python 3.7.11
- PyTorch 1.10.1
- Torchvision 0.11.2

## Intallation
### Prepare
1. Clone the repository
```
git clone https://github.com/ychsiao0809/NYCU-VRDL.git
cd 21LPCVC-UAV_VIdeo_Track-Sample-Solution
```
2. Set up environment
```
conda create --name VRDL_hm3 python=3.9
conda activate VRDL_hm3
conda install pytorch torchvision -c pytorch
```
3. Install detectron2
- Follow the tutorial of detectron2's [installation](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

1. Install dependencies:
```
pip install -r requirements.txt
```

### Data preparation
Use code below to install training and testing data:
```
!gdown https://drive.google.com/uc?id=1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG -O dataset.zip
!unzip dataset.zip > /dev/null
```

### Download Pre-trained Models
The trained weight of ResNet152 is provided [here](https://drive.google.com/file/d/1Wm7GBlxQWdcn2D5pSzM6TFUFbEUl0jz_/view?usp=sharing).

### Folder structure
```
Homework3/
├── answer.json
├── config.yaml
├── dataset/
│   ├── test/
│   │   ├── TCGA-50-5931-01Z-00-DX1.png
│   │   ├── TCGA-A7-A13E-01Z-00-DX1.png
│   │   └── ...
│   ├── test_img_ids.json
│   └── train/
│       ├── TCGA-18-5592-01Z-00-DX1/
│       │   ├── images/
│       │   │   └── TCGA-18-5592-01Z-00-DX1.png
│       │   └── masks/
│       │       ├── mask_0001.png
│       │       ├── mask_0002.png
│       │       └── ...
│       ├── TCGA-21-5784-01Z-00-DX1/
│       │   ├── images/
│       │   │   └── TCGA-21-5784-01Z-00-DX1.png
│       │   └── masks/
│       │       ├── mask_0001.png
│       │       ├── mask_0002.png
|       |       └── ...
│       └── ...
├── dataset.zip
├── detectron2/
│   └── ...
├── nucleus.py
├── output/
│   ├── last_checkpoint
│   ├── metrics.json
│   └── model_final.pth
├── preprocess_mask.py
├── README.md
├── requirements.txt
├── inference.py
└── train.py
```

## Usage
### Training
```
usage: python train.py
```
The latest checkpoint and weight of model will be stored at `output`
### Inference
```
usage: inference.py [model path]

optional arguments:
  --model_path            model path(s)
```

## Experiment
### Result (epoch=100, batch_size=32)

## Reference
- Mask R-CNN 
  - Paper - https://arxiv.org/abs/1703.06870
  - Github - https://github.com/matterport/Mask_RCNN
- Detectron
  - Github - https://github.com/facebookresearch/detectron2
  - Doc - https://detectron2.readthedocs.io/en/latest/tutorials/index.html