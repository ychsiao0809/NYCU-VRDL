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

## Dataset
The given dataset contains 291 high resolution images for training and 13 low resolution images for testing.
```
rm -rf data
mkdir -p data
gdown https://drive.google.com/uc?id=1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O "data/datasets.zip"
unzip -q data/datasets.zip -d data/ && rm data/datasets.zip
cp data/testing_lr_images/testing_lr_images/* data/testing_lr_images/
cp data/training_hr_images/training_hr_images/* data/training_hr_images/
rm -rf data/training_hr_images/training_hr_images/ data/testing_lr_images/testing_lr_images/
```

### Preprocess
The size of training image must be larger than 96x96. Run `image_prerocess.py` to remove the picture which is smaller than the required image size.
```  
python image_preprocess.py
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
├── models/
│   └── best.pth
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
cp train.py KAIR/
cd KAIR && python train.py --opt ../options/train_swinir_sr_classical.json
```
The weight of model will be stored at `superresolution/swinir_sr_classical_patch48_x3/models/`.
There are no pretrained model used in this project.

## Inference
The model weight should be place under `./model/`.
The best model weight was stored as `./model/best.pth`.
```
cp test.py KAIR/
cd KAIR && python test.py \
            --task classical_sr \
            --scale 3 \
            --training_patch_size 48 \
            --model_path ../models/best.pth \
            --folder_lq ../data/testing_lr_images/
```

The inference result would be stored at `KAIR/results/swinir_classical_sr_x3/`.

### Generate Result
To generate the result zip file, run the following command:
```
cd KAIR/results/swinir_classical_sr_x3; zip result.zip ./*.png 
mv KAIR/results/swinir_classical_sr_x3/result.zip .
```

## Experiment Result

Model | Training Epoch | PSNR
:--: | :--: | --
**Baseline** | N/A | 27.4162
SwinIR_20000_E  | 20000  | 28.1878
SwinIR_60000_G  | 60000  | 28.3503
SwinIR_60000_E  | 60000  | 28.3627
SwinIR_80000_E  | 80000  | 28.2641
SwinIR_100000_E | 100000 | 28.1895

## Reference
- SwinIR: Image Restoration Using Swin Transformer
  - [Paper](https://arxiv.org/abs/2108.10257)
  - [Github](https://github.com/JingyunLiang/SwinIR#Testing)
- KAIR - Training and Testing Codes for USRNet, DnCNN, FFDNet, SRMD, DPSR, MSRResNet, ESRGAN, BSRGAN, SwinIR
  - [Github](https://github.com/cszn/KAIR)
- Paper With Code - [Image Super-Resolution](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling)
