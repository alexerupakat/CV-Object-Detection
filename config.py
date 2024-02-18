from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "path/to/file"
__C.YOLO.ANCHORS                = "path/to/file"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.35
__C.YOLO.UPSAMPLE_METHOD        = "resize"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "path/to/file/train.txt"
__C.TRAIN.TRAIN_LOG_DIR         = "path/to/file/log.txt"
__C.TRAIN.BATCH_SIZE            = 4
__C.TRAIN.INPUT_SIZE            = 736
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.TOTAL_EPOCHS          = 200
__C.TRAIN.SAVE_WEIGHT_DIR       = "path/to/file/ckpt/"
__C.TRAIN.LEARN_RATE            = 0.00005
__C.TRAIN.WARMUP_EPOCHS         = 2

# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "path/to/file/test.txt"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZE             = 640
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "path/to/file/images/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "path/to/file"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45
