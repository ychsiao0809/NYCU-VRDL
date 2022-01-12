NYCU VRDL Homework4: Image Super Resolution
# SwinIR: Image Restoration Using Swin Transformer

## Environment
- Python 3.7.11
- PyTorch>=1.7.0
- Torchvision>=0.8.1

## Preparation
### Install
1. Clone the repository
```
git clone --recurse-submodules https://github.com/ychsiao0809/NYCU-VRDL.git
cd NYCU-VRDL/Homework4
```
2. Install dependencies:
```
pip install -r requirements.txt
cd KAIR && pip install -r requirements.txt 
```
3. Download pretrained model weigth [here]()

## Dataset
The giving dataset contains 291 high resolution images for training and 13 low resolution images for testing.
```
mkdir data
gdown https://drive.google.com/uc?id=1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O "data/dataset.zip"
unzip -q "data/dataset.zip" -d "data/" && rm "data/dataset.zip"
mv "data/testing_lr_images/testing_lr_images/*" "data/testing_lr_images/"
mv "data/training_hr_images/training_hr_images/*" "data/training_hr_images/"
rm -rf data/training_hr_images/training_lr_images/ data/testing_lr_images/testing_lr_images/
```

### Preprocess
The size of training image must be larger than 96x96. Run `image_prerocess.py` to remove the picture which is smaller than the required image size.
```  
python image_preprocess.py
```

Move the model configuration file to specified directory.
```
mv train_swinir_sr_customized.json KAIR/options/swinir/
```

### File structure
The file structure of this project should show as below.
```
Homework4/
├── data/
│   ├── testing_lr_images/
│   │   ├── 00.png
│   │   ├── ...
│   │   └── 13.png
│   └── training_hr_images/
│       ├── 2092.png
│       │   ...
│       └── tt27.png
├── KAIR/
│   ├── ...
│   ├── results/
│   │   └── swinir_classical_sr_x3
│   │       ├── 00_pred.png
│   │       ├── ...
│   │       └── 13_pred.png
│   ├── ...
│   ├── superresolution/
│   │   └── swinir_sr_classical_patch48_x3/
│   │       ├── images/
│   │       ├── models/
│   │       │   ├── 10000_E.pth
│   │       │   ├── 10000_G.pth
│   │       │   ├── 10000_optimizerG.pth
│   │       │   ├── ...
│   │       │   └── best.pth
│   │       └── options/
│   ├── ...
│   └── requirements.txt
├── options/
│   └── train_swinir_sr_customized.json
├── train.py
├── test.py
├── image_preprocess.py
├── Makefile
├── README.md
├── requirements.txt
└── result.zip
```
- `data/training_hr_`, `datasets/test/`: origin datasets
- `KAIR/options/swinir/train_swinir_sr_customized.json`: customized configuration file for training
- `image_preprocess.py`: modified yolo detection
- `train.py`: training code modified from `main_train_psnr.py` [KAIR](https://github.com/cszn/KAIR)
- `test.py`: testing code modified from `main_test_swinir.py` in [KAIR](https://github.com/cszn/KAIR)

## Train
```
mv train.py KAIR/
cd KAIR && python train.py --opt ../options/train_swinir_sr_classical.json
```
There are no pretrained model used in this project.

## Inference
The model weight should be place under `superresolution/swinir_sr_classical_patch48_x3/models/`. The best model weight could be installed at [here]().
```
mv test.py KAIR/
cd KAIR && python test.py --task classical_sr --scale 3 --training_patch_size 48 --model_path superresolution/swinir_sr_classical_patch48_x3/models/[model weight].pth --folder_lq ../data/testing_lr_images/
```

## Reference
- SwinIR: Image Restoration Using Swin Transformer
  - [Paper](https://arxiv.org/abs/2108.10257)
  - [Github](https://github.com/JingyunLiang/SwinIR#Testing)
- KAIR - Training and Testing Codes for USRNet, DnCNN, FFDNet, SRMD, DPSR, MSRResNet, ESRGAN, BSRGAN, SwinIR
  - [Github](https://github.com/cszn/KAIR)
- Paper With Code - [Image Super-Resolution](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling)
