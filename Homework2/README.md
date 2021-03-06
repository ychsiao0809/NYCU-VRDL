NYCU VRDL Homework2: Object Detection on Street View House Numbers
# YOLOv5

## Environment
- Python 3.7.11
- PyTorch>=1.7.0
- Torchvision>=0.8.1
- h5py 3.6.0

## Install
### Prepare
1. Clone the repository
```
git clone --recurse-submodules https://github.com/ychsiao0809/NYCU-VRDL.git
cd Homework2
```
2. Install dependencies:
```
pip install -r requirements.txt
pip install -r yolov5/requirements.txt 
```
3. Download pretrained model weigth [here](https://github.com/ychsiao0809/NYCU-VRDL/blob/main/Homework2/weights/yolov5m/best.pt)

## Dataset
The giving SVHN dataset contains 33402 images for training and 13068 images for testing. This project uses the YOLOv5 pre-trained model to fix this challenge.

See [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

## Data Preprocess
Run `prerocess.py` tp extract label of data from `digitStruct.mat`  and split train data into train and validation.
```
usage: preprocess.py [-h] [-v VALID]

optional arguments:
  -h, --help            show this help message and exit
  -v VALID, --valid VALID
                        number of validation data
```
### File structure
The file structure should look like below.
```
/
├── datasets/
│   ├── svhn/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   │   ├── 10.png
│   │   │   │   │   ...
│   │   │   │   └── 9999.png
│   │   │   ├── labels/
│   │   │   │   ├── 10.txt
│   │   │   │   │   ...
│   │   │   │   └── 9999.txt
│   │   │   └── labels.cache
│   │   └── valid/
│   │       ├── images/
│   │       │   ├── 1.png
│   │       │   │   ...
│   │       │   └── 9992.png
│   │       ├── labels/
│   │       │   ├── 1.txt
│   │       │   │   ...
│   │       │   └── 9992.txt
│   │       └── labels.cache
│   ├── test/
│   │   ├── 100009.png
│   │   │   ...
│   │   └── 99942.png
│   ├── train/
│   │   ├── 1.png
│   │   │   ...
│   │   ├── 9999.png
│   │   ├── digitStruct.mat
│   │   └── see_bboxes.m
├── detect.py
├── inference.ipynb
├── preprocess.py
├── README.md
├── requirements.txt
├── answer.json
├── toJSON.py
└── yolov5/
    └── ...
```
- `datasets/train/`, `datasets/test/`: origin datasets
- `datasets/svhn/`: Randomly split origin train data into training and validation dataset and store their label under `labels/`. See `preprocess.py`.
- `preprocess.py`: train data preprocessing
- `detect.py`: modified yolo detection
- `toJSON.py`: transform yolov5 data label form into json
- `yolo`: See [YOLOv5 github page](https://github.com/ultralytics/yolov5)

## Train
Make sure
```
```
Train model with pretrained weights, see [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
```
cd yolov5
python train.py --img 320 --batch 15 --epochs 50 --data svhn.yaml --weight yolov5m.pt
```

## Validation result
- yolov5m
```
50 epochs completed in 2.666 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 42.2MB
Optimizer stripped from runs/train/exp/weights/best.pt, 42.2MB

Validating runs/train/exp/weights/best.pt...
Fusing layers...
Model Summary: 290 layers, 20889303 parameters, 0 gradients, 48.1 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 100/100 [00:12<00:00,  8.00it/s]                                                                    
                 all       3000       6580      0.944      0.941      0.956      0.511
                   0       3000        440      0.937      0.939       0.95      0.529
                   1       3000       1272      0.925      0.907      0.929      0.413
                   2       3000        931      0.949      0.968      0.971      0.531
                   3       3000        753      0.946      0.937      0.954      0.517
                   4       3000        661      0.939      0.953      0.956      0.505
                   5       3000        637      0.962      0.943      0.962      0.525
                   6       3000        516      0.949      0.931      0.956      0.527
                   7       3000        486      0.956      0.942      0.966      0.525
                   8       3000        492      0.948      0.957      0.962      0.525
                   9       3000        392      0.933      0.931      0.955      0.517
```
- yolov5l
```
50 epochs completed in 4.873 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 92.8MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 92.8MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers...
Model Summary: 367 layers, 46156743 parameters, 0 gradients, 107.9 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 100/100 [00:15<00:00,  6.54it/s]
                 all       3000       6580      0.946      0.942      0.957      0.513
                   0       3000        440      0.944      0.948       0.95      0.527
                   1       3000       1272      0.922      0.912      0.923      0.411
                   2       3000        931      0.948      0.964      0.971      0.532
                   3       3000        753      0.957      0.942      0.959       0.52
                   4       3000        661      0.948      0.964       0.96      0.511
                   5       3000        637      0.961      0.931      0.958      0.534
                   6       3000        516      0.936      0.938      0.955       0.53
                   7       3000        486      0.946      0.937      0.965      0.521
                   8       3000        492      0.955       0.95       0.97      0.532
                   9       3000        392       0.94      0.934      0.955      0.517
```

### Run inference and benchmark
- `test`: number of input for inference speed time
```
python detect.py --source ../datasets/svhn/train/images/ --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name exp --test 100
```

### Detect
#### Run testing
`--source`: can be assigned with directory or specified file
`--weights`: path of model weights
`--conf`: confidence threshold
`--save-txt`: save label result
`--save-txt`: save confidence result (while `--save-txt` is `True`)
`--name`: name of experiment directory
```
python detect.py --source ../datasets/test --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name [experiment name]
```

#### Turn YOLOv5 labels to JSON
```
python toJSON.py --exp [experiment name]
```
#### Output
`answer.json`
```
[
    {
        "image_id": 117,
        "category_id": 3,
        "bbox": [
            41.999961,
            8.999991999999999,
            12.00005,
            24.999988000000002
        ],
        "score": 0.71503
    },
    {
        "image_id": 117,
        "category_id": 8,
        "bbox": [
            53.999957499999994,
            11.000002999999998,
            13.999987,
            25.999982
        ],
        "score": 0.770699
    },
    .
    .
    .
]   
```

## Result
| pretrained model | validation mAP | layers | parameters | test mAP |
|:--:|:--:|:--:|:--:|:--:|
| yolov5m	| 0.51171 | 290 | 20889303 | 0.41240 |
| yolov5l	| 0.51342 | 367 | 46156743 | 0.41240 |

## Reference
- [h5py guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
