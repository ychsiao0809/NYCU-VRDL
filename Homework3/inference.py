import os
import copy
import torch
import warnings
import argparse
from detectron2.engine import DefaultPredictor, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2 import model_zoo

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from nucleus import *

def main(args):
    """ register dataset """
    for d in ['train', 'val']:
        DatasetCatalog.register('nucleus_' + d, lambda d = d: get_nucleus_datasets(d))
        MetadataCatalog.get('nucleus_' + d).set(thing_classes=['nucleus'])

    nucleus_metadata = MetadataCatalog.get("nucleus_train")
    nucleus_metadata = MetadataCatalog.get("nucleus_val")
    
    """ main function """
    cfg = get_cfg()
    model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"    
    # model = "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"
    modelfile = model_zoo.get_config_file(model)
    cfg.merge_from_file(modelfile)
    # cfg.merge_from_file('config.yaml')
    cfg.DATASETS.TRAIN = ("nucleus_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    # cfg.DATASETS.TRAIN = ("VOC_dataset",)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize trainer and train
    # trainer = Trainer(cfg)
    cfg.MODEL.WEIGHTS = os.path.join(args.model_path)
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    # output  COCO style evaluation results
    """
    [{
        "image_id": int, 
        "category_id": int, 
        "segmentation": RLE, 
        "score": float,
    }]
    """
    coco_result = []
    dataset_dicts = get_nucleus_datasets(mode="test")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        image_id = d["image_id"]
        category_id = 0
        print(outputs["instances"].scores.shape[0])
        for i in range(outputs["instances"].scores.shape[0]):
            segmentation = encode(np.array(outputs["instances"].pred_masks[i].cpu(), order="F"))
            segmentation["counts"] = str(segmentation["counts"],'utf-8')
            score = outputs["instances"].scores[i].item()
            coco_result.append({"image_id": image_id, "category_id": category_id, "segmentation": segmentation, "score": score})
    print('open')
    with open("answer.json", "w") as f:
        json.dump(coco_result, f)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, default= 'output/model_final.pth', help='model path(s)')
    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    # Set Environment
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    num_gpus = 1
    batch_size = 2 * num_gpus

    # Initialize args and data
    args = default_argument_parser().parse_args()
    setup_logger()
    opt = parse_opt()

    main(opt)
    # launch(
    #     main(opt),
    #     num_gpus_per_machine=num_gpus,
    #     num_machines=1,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(),
    # )