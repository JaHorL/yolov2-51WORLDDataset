import sys
sys.path.append("../")
import os
import cv2
from glob import glob
from utils import math
from utils import vis_tools
from config.config import cfg
from data import postprocess
from data import loader
import numpy as np  


img_pred_files      = glob(cfg.YOLOv2.LOG_DIR+"/pred/img_pred/*")
img_anchors         = loader.load_anchors(cfg.IMG.ANCHORS)
img_dir             = os.path.join(cfg.YOLOv2.DATASETS_DIR, "image_files/")


for fi in img_pred_files:
	img_pred = np.load(fi)
	img_map = img_pred.reshape([cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, 6, 11])
	vis_tools.imshow_img(np.max(math.sigmoid(img_map[..., 0]), axis=-1))
	vis_tools.imshow_img(np.max(math.sigmoid(img_map[..., 1:cfg.YOLOv2.CLASSES_NUM+1])[..., 0], axis=-1))
	img_bboxes = postprocess.parse_img_predmap(img_pred, img_anchors)
	img_bboxes = postprocess.img_nms(img_bboxes, cfg.IMG.IOU_THRESHOLDS)
	img_file = img_dir + fi[-14:-8] + ".png"  
	img = cv2.imread(img_file)
	vis_tools.imshow_img_bbox(img, np.array(img_bboxes))
