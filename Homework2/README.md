## Train
```
python train.py --img 320 --batch 15 --epochs 50 --data svhn.yaml --weight yolov5m.pt
```
## Validation result
yolov5m
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

yolov5l
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

## Run inference and benchmark
```
python detect.py --source ../datasets/svhn/train/images/ --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name exp --test 100
```

## Detect
### Usage
```
usage: detect.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE]
                 [--imgsz IMGSZ [IMGSZ ...]] [--conf-thres CONF_THRES]
                 [--iou-thres IOU_THRES] [--max-det MAX_DET] [--device DEVICE]
                 [--view-img] [--save-txt] [--save-conf] [--save-crop]
                 [--nosave] [--classes CLASSES [CLASSES ...]] [--agnostic-nms]
                 [--augment] [--visualize] [--update] [--project PROJECT]
                 [--name NAME] [--exist-ok] [--line-thickness LINE_THICKNESS]
                 [--hide-labels] [--hide-conf] [--half] [--dnn] [--test TEST]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS [WEIGHTS ...]
                        model path(s)
  --source SOURCE       file/dir/URL/glob, 0 for webcam
  --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
                        inference size h,w
  --conf-thres CONF_THRES
                        confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     maximum detections per image
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --view-img            show results
  --save-txt            save results to *.txt
  --save-conf           save confidences in --save-txt labels
  --save-crop           save cropped prediction boxes
  --nosave              do not save images/videos
  --classes CLASSES [CLASSES ...]
                        filter by class: --classes 0, or --classes 0 2 3
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --visualize           visualize features
  --update              update all models
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
  --exist-ok            existing project/name ok, do not increment
  --line-thickness LINE_THICKNESS
                        bounding box thickness (pixels)
  --hide-labels         hide labels
  --hide-conf           hide confidences
  --half                use FP16 half-precision inference
  --dnn                 use OpenCV DNN for ONNX inference
  --test TEST           inference speed test

```
### Example
```
python detect.py --source ../datasets/test --weights runs/train/exp/weights/best.pt --conf 0.25 --save-txt --save-conf --name [experiment name]
```

### Turn YOLOv5 labels to JSON
```
python toJSON.py --exp [experiment name]
```

## Result
| pretrained model | layers | parameters | mAP |
|:--:|:--:|:--:|:--:|
| yolov5m	| 290 | 20889303 | 0.41240 |
| yolov5l	| 367 | 46156743 | 0.41240 |

## Reference
- [h5py guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
