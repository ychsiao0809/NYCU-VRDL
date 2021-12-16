import os
import copy
import torch
import warnings
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
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


def main():
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
    cfg.DATASETS.TEST = ("nucleus_val")
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
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    # Train with {num_gpus} GPUs
    # main()
    # Set Environment
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    num_gpus = 1
    batch_size = 2 * num_gpus

    # Initialize args and data
    args = default_argument_parser().parse_args()
    setup_logger()

    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(),
    )