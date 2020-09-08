import sys
sys.path.append("../")
import os
import cv2
from data import dataset
from utils import utils
from utils import vis_tools
from tqdm import tqdm
from config.config import cfg
from data import postprocess
import numpy as np
from utils import math


def get_idx(array):
    idx_tuple = np.where(array==1)
    u, idx = np.unique(idx_tuple[0], return_index = True)
    return u, idx_tuple[1][idx]


def parse_img_labelmap(predmap, anchors):
    anchor_shape = [cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, anchors.shape[0], anchors.shape[1]]
    anchors = np.broadcast_to(np.array(anchors), anchor_shape)
    h = np.tile(np.array(range(cfg.IMG.OUTPUT_H))[:, np.newaxis], [1, cfg.IMG.OUTPUT_W])
    w = np.tile(np.array(range(cfg.IMG.OUTPUT_W))[np.newaxis, :], [cfg.IMG.OUTPUT_H, 1])
    hw_grid = np.stack((h, w), axis=-1)
    hw_shape = [cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.ANCHORS_NUM, 2]
    hw_grid = np.tile(hw_grid, cfg.IMG.ANCHORS_NUM).reshape(hw_shape) 
    box_shape = [cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.ANCHORS_NUM, cfg.YOLOv2.CLASSES_NUM+cfg.IMG.BBOX_DIM+1]
    predmap = predmap.reshape(box_shape)
    predmap = np.concatenate((predmap, hw_grid, anchors), axis=-1)
    preds = predmap[predmap[..., 0]>0.3]
    bbox = preds[..., cfg.YOLOv2.CLASSES_NUM+1:]
    objness = preds[..., 0][..., np.newaxis]
    clsness = preds[..., 1:cfg.YOLOv2.CLASSES_NUM+1]
    prob = objness * clsness
    cls_max_prob = np.max(prob, axis=-1)
    cls_idx = np.argmax(prob, axis=-1)
    bbox[..., :2] = bbox[..., :2]
    bbox[..., 2:4] = bbox[..., 2:4]
    x = (bbox[..., 0] + bbox[..., -4]) * cfg.IMG.STRIDE / cfg.IMG.H_SCALE_RATIO 
    y = (bbox[..., 1] + bbox[..., -3]) * cfg.IMG.STRIDE / cfg.IMG.W_SCALE_RATIO
    h = bbox[..., 2] / cfg.IMG.H_SCALE_RATIO * bbox[:, -2]
    w = bbox[..., 3] / cfg.IMG.W_SCALE_RATIO * bbox[:, -1]
    left = y - w / 2
    top = x - h / 2
    right = y + w / 2
    bottom = x + h / 2
    result = np.stack([cls_idx, cls_max_prob, left, top, right, bottom], axis=-1)
    return result[cls_max_prob>0.3]

trainset            = dataset.Dataset('train')
img_anchors         = trainset.img_anchors
img_dir             = os.path.join(cfg.YOLOv2.DATASETS_DIR, "image_files/")


for j in range(len(trainset)):
    data = trainset.load()
    # vis_tools.imshow_img(data[0][0].astype(np.float32))
    shape = [cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.LABEL_Z]
    label = np.reshape(data[1][0], shape)
    imglabel = parse_img_labelmap(label, img_anchors)
    img_bboxes = postprocess.img_nms(imglabel, cfg.IMG.IOU_THRESHOLDS)
    vis_tools.imshow_img_bbox(data[0][0].astype(np.float32), img_bboxes)
