import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C                               = edict()
cfg                               = __C

__C.YOLOv2                        = edict()
__C.YOLOv2.DATASET_TYPE           = "simone"
__C.YOLOv2.CLASS_DICT             = {'Car':0,'Rider':4,'TrafficLight':5,'Truck':1,'Bus':2,'Pedestrian':3,'SpecialVehicle':0, 'SpeedLimitSign':5}
__C.YOLOv2.CLASSES_COLOR          = [(255,0,0),(255,255,0),(255,0,255),(0,255,0),(128,64,255),(0,255,255)]
__C.YOLOv2.CLASSES_NUM            = len(__C.YOLOv2.CLASSES_COLOR)
__C.YOLOv2.EPSILON                = 0.00001
__C.YOLOv2.MAX_PTS_NUM            = 200000
__C.YOLOv2.ROOT_DIR               = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
__C.YOLOv2.LOG_DIR                = osp.join(__C.YOLOv2.ROOT_DIR, 'logs')
__C.YOLOv2.DATASETS_DIR           = "/media/jhli/57DF22050921ED01/exchange/dl_dataset/image/" + __C.YOLOv2.DATASET_TYPE 
__C.YOLOv2.TRAIN_DATA             = osp.join(__C.YOLOv2.DATASETS_DIR, "training.txt")
__C.YOLOv2.VAL_DATA               = osp.join(__C.YOLOv2.DATASETS_DIR, "val.txt")
__C.YOLOv2.TEST_DATA              = osp.join(__C.YOLOv2.DATASETS_DIR, "testing.txt")
__C.YOLOv2.MOVING_AVE_DECAY       = 0.9995
__C.YOLOv2.IS_USE_THREAD          = True
__C.YOLOv2.POINTS_THRESHOLDS      = [20,40,40,10,10,5]
__C.YOLOv2.PRINTING_STEPS         = 20
__C.YOLOv2.LIDAR_HEIGHT           = 1.8


__C.IMG                     = edict()
__C.IMG.ANCHORS             = __C.YOLOv2.ROOT_DIR + "/src/config/anchors/image_anchors.txt"
__C.IMG.LOSS_SCALE          = np.array([1.00, 1.00, 1.0, 1.0, 1.0, 1.0])
__C.IMG.INPUT_H             = 1080
__C.IMG.INPUT_W             = 1920
__C.IMG.H_SCALE_RATIO       = __C.IMG.INPUT_H / 1080
__C.IMG.W_SCALE_RATIO       = __C.IMG.INPUT_W / 1920
__C.IMG.BBOX_DIM            = 4
__C.IMG.STRIDE              = 16
__C.IMG.IS_IMG_AUG          = False
__C.IMG.OUTPUT_H            = int(__C.IMG.INPUT_H / __C.IMG.STRIDE)
__C.IMG.OUTPUT_W            = int(__C.IMG.INPUT_W / __C.IMG.STRIDE)
__C.IMG.ANCHORS_NUM         = 6
__C.IMG.PER_ANCHOR_DIM      = (__C.IMG.BBOX_DIM + 1 + __C.YOLOv2.CLASSES_NUM)
__C.IMG.LABEL_Z             = int((__C.IMG.BBOX_DIM + 1 + __C.YOLOv2.CLASSES_NUM) * __C.IMG.ANCHORS_NUM)
__C.IMG.IOU_THRESHOLDS      = [0.5, 0.5, 0.5, 0.3, 0.3, 0.3]
__C.IMG.OUTPUT_Z            = __C.IMG.ANCHORS_NUM * (cfg.IMG.BBOX_DIM + cfg.YOLOv2.CLASSES_NUM + 1)
__C.IMG.IS_DRAW_PROB        = False    


__C.TRAIN                     = edict()

__C.TRAIN.PRETRAIN_WEIGHT     = ""
__C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.SAVING_STEPS        = int(4000 / __C.TRAIN.BATCH_SIZE)
__C.TRAIN.FRIST_STAGE_EPOCHS  = 0
__C.TRAIN.SECOND_STAGE_EPOCHS = 10
__C.TRAIN.WARMUP_EPOCHS       = 0
__C.TRAIN.LEARN_RATE_INIT     = 1e-4
__C.TRAIN.LEARN_RATE_END      = 1e-6
__C.TRAIN.IS_DATA_AUG         = True


__C.EVAL                      = edict()
__C.EVAL.BATCH_SIZE           = 1
__C.EVAL.WEIGHT               = ""
__C.EVAL.OUTPUT_GT_PATH       = osp.join(__C.YOLOv2.LOG_DIR, "gt")
__C.EVAL.OUTPUT_PRED_PATH     = osp.join(__C.YOLOv2.LOG_DIR, "pred")

if __name__ == "__main__":
  print(cfg)  
