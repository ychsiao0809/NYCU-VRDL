NYCU VRDL Homework1: Bird Classification
# Deep Residual Learning for Image Recognition
Official PyTorch model implement based on paper [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)
## Environment
- Python 3.7.11
- PyTorch 1.5.1
- Torchvision 0.6.1

## Intallation
### Prepare
1. Clone the repository
```
git clone https://github.com/ychsiao0809/NYCU-VRDL.git
cd 21LPCVC-UAV_VIdeo_Track-Sample-Solution
```
2. Install dependencies:
```
pip install -r requirements.txt
```
### Download Pre-trained Models
The trained weight of ResNet152 is provided [here](https://drive.google.com/file/d/1-7rayKLTUCdu6GhpOGThwBmNwGkMT-qx/view?usp=sharing).

### Folder structure
Training and Testing data should be placed at `data/train` and `data/test`
```
Homework1
├── answer.txt
├── best.ckpt
├── data/
│   ├── answer.txt
│   ├── classes.txt
│   ├── test/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   ├── 0004.jpg
│   │   ├── ...
│   ├── testing_img_order.txt
│   ├── train/
│   │   ├── 0003.jpg
│   │   ├── 0008.jpg
│   │   ├── 0010.jpg
│   │   ├── ...
│   └── training_labels.txt
├── inference.py
├── README.md
├── requirements.txt
└── train.py
```

## Usage
### Training
```
usage: train.py [-h] [-i INPUT] [-o OUTPUT] [-b BATCH] [-e EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        load pretrained model state file
  -o OUTPUT, --output OUTPUT
                        output model state file name
  -b BATCH, --batch BATCH
                        batch size.
  -e EPOCH, --epoch EPOCH
                        number of training epoches.
```
### Inference
```
usage: inference.py [-h] -i INPUT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        load model state file
```

## Experiment
### Result (epoch=100, batch_size=32)
|Model|Validation Accuracy|Test Accuracy|Learning Rate|Weight Decay|FC|
|:--:|:--:|:--:|:--:|:--:|:--:|
|ResNet50|0.49|0.420046|0.001|0.01|2048 x 200|
|ResNet50|0.503333|0.43785|0.001|0.01|2048 x 512 x 200|
|ResNet50|0.563333|0.420046|0.001|0.05|2048 x 512 x 200|
|ResNet50 (SGD)|0.596667|0.513353|0.01|N/A|2048 x 512 x 200|
|ResNet152 (SGD)|0.63| N/A |0.01| N/A |2048 x 512 x 200|

### Reference
- ResNet
  - https://arxiv.org/abs/1512.03385
- ResNet50 model
  - https://pytorch.org/vision/stable/models.html
- AdamW (optimizer)
  - https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- Model Training and Validation Code
  - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- Early stopping
  - https://clay-atlas.com/blog/2020/09/29/pytorch-cn-early-stopping-code/