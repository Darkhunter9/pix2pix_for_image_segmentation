# manage config files according to https://github.com/rbgirshick/yacs
from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 42

_C.DATASET_ROOT = "./data/cityscapes"
_C.NUM_CLASS = 20

_C.MODEL = "unet"
_C.MODEL_ENCODER = "resnet50"

_C.SHAPE = (1024, 2048) # training and validation images are resized to this shape first
_C.CROP_SIZE = (512, 512) # after resizing, training images are random cropped into this size
_C.TRAIN_CROP = True
_C.VAL_CROP = False
_C.TRAIN_REPEAT = 2 # the number of times training set is repeated in each epoch
_C.SUBSET = False # for debug purpose
_C.SUBSET_SIZE = 8 # number of images in subset

_C.EPOCH = 50
_C.BATCH_SIZE = 8
_C.OPTIMIZER = "AdamW"
_C.LR = 1e-4 # initial learning rate
_C.LR_SCHEDULER = "CosineAnnealingWarmRestarts" # learning rate scheduler
_C.LR_MIN = 1e-7 # minimum learning rate
_C.T_RESTART = 500 # number of iterations for restart

_C.NAME_SUFFIX = ""
_C.SAVE = False # save model

_C.GAN = CN()
_C.GAN.WITH_SRC = False
_C.GAN.LAMBDA = 1
_C.GAN.LAMBDA_DIS = 0.1
_C.GAN.DIS_ITER = 3

def get_config(yaml_path):
    cfg = _C.clone()
    cfg.merge_from_file(yaml_path)
    cfg.freeze()
    print(cfg)
    return cfg
