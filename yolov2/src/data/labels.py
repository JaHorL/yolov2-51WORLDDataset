import cv2
import numpy as np
import math
from config.config import cfg


def create_img_label(image_annos, anchors):
    def iou_wh(r1, r2):
        min_w = min(r1[0],r2[0])
        min_h = min(r1[1],r2[1])
        area_r1 = r1[0]*r1[1]
        area_r2 = r2[0]*r2[1]	
        intersect = min_w * min_h		
        union = area_r1 + area_r2 - intersect
        return intersect/union

    def get_active_anchors(roi, anchors):
        indxs = []
        iou_max, index_max = 0, 0
        for i,a in enumerate(anchors):
            iou = iou_wh(roi, a)
            if iou>0.5:
                indxs.append(i)
            if iou > iou_max:
                iou_max, index_max = iou, i
        if len(indxs) == 0:
            indxs.append(index_max)
        return indxs

    obj_num = len(image_annos)
    label = np.zeros([cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.ANCHORS_NUM, 
                      cfg.IMG.PER_ANCHOR_DIM], dtype=np.float32)
    for i in range(obj_num):
        box2d_corners = image_annos[i]['bbox']
        obj_type = cfg.YOLOv2.CLASS_DICT[image_annos[i]['type']]
        h = (box2d_corners[3]-box2d_corners[1]) * cfg.IMG.H_SCALE_RATIO
        w = (box2d_corners[2]-box2d_corners[0]) * cfg.IMG.W_SCALE_RATIO 
        center_h = (box2d_corners[3]+box2d_corners[1])/2 * cfg.IMG.H_SCALE_RATIO
        center_w = (box2d_corners[2]+box2d_corners[0])/2 * cfg.IMG.W_SCALE_RATIO
        grid_h = int(center_h / cfg.IMG.STRIDE)
        grid_w = int(center_w / cfg.IMG.STRIDE)
        grid_h_offset = center_h / cfg.IMG.STRIDE - grid_h
        grid_w_offset = center_w / cfg.IMG.STRIDE - grid_w
        active_idxs = get_active_anchors([h, w], anchors)
        for idx in active_idxs:
            dh = h / anchors[idx][0]
            dw = w / anchors[idx][1]
            label[grid_h, grid_w, idx, 0] = 1
            label[grid_h, grid_w, idx, 1+int(obj_type)] = 1
            label[grid_h, grid_w, idx, -4:] = np.array([grid_h_offset, grid_w_offset, dh, dw])
    return label

if __name__ == "__main__":
    pass