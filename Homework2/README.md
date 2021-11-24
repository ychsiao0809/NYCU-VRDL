## Train
```
python train.py --img 320 --batch 15 --epochs 50 --data svhn.yaml --weight yolov5m.pt
```
## Validation result
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
## Run inference and benchmark
```
python detect.py --source ../datasets/svhn/train/images/ --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name exp --test 100
```

## Detect
```
python detect.py --source ../datasets/test --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name exp
```

## Result
| pretrained model | mAP |
|:--:|:--:|
| yolov5m	| 0.41240 |

## Reference
- [h5py guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- 
