import os
import shutil
import numpy as np
import cv2
from utils import vis_tools
from utils import utils
from utils import math
from config.config import cfg




def parse_img_predmap(predmap, anchors):
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
    preds = predmap[math.sigmoid(predmap[..., 0])>0.1]
    bbox = preds[..., cfg.YOLOv2.CLASSES_NUM+1:]
    objness = math.sigmoid(preds[..., 0])[..., np.newaxis]
    clsness = math.sigmoid(preds[..., 1:cfg.YOLOv2.CLASSES_NUM+1])
    prob = objness * clsness
    cls_max_prob = np.max(prob, axis=-1)
    cls_idx = np.argmax(prob, axis=-1)
    bbox[..., :2] = math.sigmoid(bbox[..., :2]) 
    bbox[..., 2:4] = np.exp(bbox[..., 2:4]) 
    x = (bbox[..., 0] + bbox[..., -4]) * cfg.IMG.STRIDE / cfg.IMG.H_SCALE_RATIO 
    y = (bbox[..., 1] + bbox[..., -3]) * cfg.IMG.STRIDE / cfg.IMG.W_SCALE_RATIO
    h = bbox[..., 2] / cfg.IMG.H_SCALE_RATIO * bbox[:, -2]
    w = bbox[..., 3] / cfg.IMG.W_SCALE_RATIO * bbox[:, -1]
    left = y - w / 2
    top = x - h / 2
    right = y + w / 2
    bottom = x + h / 2
    result = np.stack([cls_idx, cls_max_prob, left, top, right, bottom], axis=-1)
    return result[cls_max_prob>0.1]


def img_nms(bboxes, iou_thresholds, sigma=0.3, method='nms'):

    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious

    classes_in_img = list(set(bboxes[:, 0]))
    best_bboxes = []
    for cls_type in classes_in_img:
        cls_mask = (bboxes[:, 0] == cls_type)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes):
            max_ind = np.argmax(cls_bboxes[:, 1])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind+1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, 2:], cls_bboxes[:, 2:])
            weight = np.ones((len(iou),), dtype=np.float32)
            assert method in ['nms', 'soft-nms']
            if method == 'nms':
                iou_mask = iou > iou_thresholds[int(cls_type)]
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 1] = cls_bboxes[:, 1] * weight
            score_mask = cls_bboxes[:, 1] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes




def cal_tpfnfp(gts, results, thresholds):
    
    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        iou          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        max_ind       = np.argmax(iou)
        return max_ind, iou[max_ind]

    tps = []
    fps = []
    fns = []
    for cls_type in range(cfg.YOLOv2.CLASSES_NUM):
        gt_bboxes = gts[gts[..., 1]==cls_type]
        result_bboxes = results[results[..., 0]==cls_type]
        for bbox in result_bboxes:
            if len(gt_bboxes) == 0: continue
            thres = thresholds[cls_type]
            max_ind, iou = bboxes_iou(bbox[..., 2:6], gt_bboxes[..., 2:6])
            obj_bbox = gt_bboxes[max_ind]
            # print(max_ind, iou)
            if iou > thres:
                tps.append([obj_bbox.tolist(), bbox.tolist()])
                gt_bboxes = np.concatenate([gt_bboxes[: max_ind], gt_bboxes[max_ind+1: ]])
            else:
                fps.append(bbox.tolist())
        for b in gt_bboxes:
            fns.append(b.tolist())
    return tps, fps, fns


def format_tps_json(tps, frame_id):
    obj_list = []
    for tp in tps:
        if len(tp) == 0: continue
        obj_dict = {}
        format_gt_json(obj_dict, tp[0])
        format_pred_json(obj_dict, tp[1])
        obj_list.append(obj_dict)
    return {str(frame_id):obj_list}


def format_fps_json(fps, frame_id):
    obj_list = []
    for fp in fps:
        if len(fp) == 0: continue
        obj_dict = {}
        format_pred_json(obj_dict, fp)
        obj_list.append(obj_dict)
    return {str(frame_id):obj_list}


def format_fns_json(fns, frame_id):
    obj_list = []
    for fn in fns:
        obj_dict = {}
        format_gt_json(obj_dict, fn)
        obj_list.append(obj_dict)
    return {str(frame_id):obj_list}


def format_gt_json(obj_dict, gt):
    obj_dict["id"] = int(gt[0])
    obj_dict["gt_type"] = int(gt[1])
    obj_dict["gt_bbox"] = gt[2:6]
    obj_dict["pixel_rate"] = gt[6]
    obj_dict["rect_rate"] = gt[7]
    return


def format_pred_json(obj_dict, result):
    obj_dict["result_type"] = int(result[0])
    obj_dict["result_prob"] = result[1]
    obj_dict["result_bbox"] = result[2:6]
    return



def format_result_as_json(gts, img_bboxes, dumped_json, frame_id):
    types, dimensions, box2d_corners, locations, rzs, ids, rates = [np.array(gt) for gt in gts]
    tp, fp, fn = [], [], []

    if len(gts[0]) == 0:
        for b in img_bboxes: 
            fp.append(b.tolist())
    else:
        gts = np.concatenate([ids[:, np.newaxis], types[:, np.newaxis], box2d_corners, rates], axis=-1)
        gts = np.around(gts, decimals=3)
        if len(img_bboxes) == 0:
            for b in gts:
                fn.append(b.tolist())
        else:
            img_bboxes = np.around(np.array(img_bboxes), decimals=3)
            tp, fp, fn = cal_tpfnfp(gts, img_bboxes, cfg.IMG.IOU_THRESHOLDS)
    dumped_json["tps"].append(format_tps_json(tp, frame_id))	
    dumped_json["fps"].append(format_fps_json(fp, frame_id))
    dumped_json["fns"].append(format_fns_json(fn, frame_id))
    return dumped_json





if __name__ == "__main__":
    pass