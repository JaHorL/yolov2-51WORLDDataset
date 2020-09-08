import cv2
import os
import numpy as np
from config.config import cfg


def draw_img_bboxes2d(img, bboxes):
    if(len(bboxes) == 0):
        return img
    for n in range(len(bboxes)):
        b = bboxes[n][2:6].astype(np.int32)
        color = cfg.YOLOv2.CLASSES_COLOR[int(bboxes[n][0])]
        cv2.line(img, (b[0],b[1]), (b[2],b[1]), color, 2, cv2.LINE_AA)
        cv2.line(img, (b[0],b[1]), (b[0],b[3]), color, 2, cv2.LINE_AA)
        cv2.line(img, (b[0],b[3]), (b[2],b[3]), color, 2, cv2.LINE_AA)
        cv2.line(img, (b[2],b[1]), (b[2],b[3]), color, 2, cv2.LINE_AA)
        if cfg.IMG.IS_DRAW_PROB:
            x = int((b[0] + b[2]) / 2)
            y = int((b[1] + b[3]) / 2)
            prob = round(bboxes[n][1], 2)
            cv2.putText(img, prob, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    return  img

def imshow_img(img, new_size=None, name=None):
        if not name:
                name = 'img1'
        if new_size:
                img = cv2.resize(img, new_size)
        cv2.imshow(name, img)
        cv2.moveWindow(name, 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def write_img(img, save_path, idx, size=None):
        name = save_path + "/" + idx + '.jpg'
        if size:
            img = img.resize(size)
        # img = img.astype(np.int32)
        cv2.imwrite(name, img)


def imshow_img_bbox(img, bboxes):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0): 
        return
    img = draw_img_bboxes2d(img, bboxes)
    imshow_img(img)


def imwrite_img_bbox(img, bboxes, save_path, idx):
    bboxes = np.array(bboxes)
    if(len(bboxes) == 0): 
        return
    corners = np.stack([bboxes[..., 2], bboxes[..., 3], bboxes[..., 4], bboxes[..., 5]], axis=-1)
    img = draw_img_bboxes2d(img, corners, bboxes[..., 0])
    write_img(img, save_path, idx)

