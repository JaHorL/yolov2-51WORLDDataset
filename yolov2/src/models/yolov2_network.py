# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf
from config.config import cfg
from models import backbone
from models import headnet
from models import loss


class YOLOv2Network(object):
    def __init__(self):
        self.img_anchor_num = 6
        self.img_output_z = self.img_anchor_num * (cfg.IMG.BBOX_DIM + cfg.YOLOv2.CLASSES_NUM + 1)


    def net(self, img_input, trainable):
        with tf.variable_scope('yolov2_backbone') as scope:
            route_1, img_block = backbone.yolov2_resnet22(img_input, trainable)
        with tf.variable_scope('yolov2_headnet') as scope:
            img_pred = headnet.yolov2_headnet(route_1, img_block, self.img_output_z, trainable)
        return img_pred, 

    def load(self):
        img_shape = [None, cfg.IMG.INPUT_H, cfg.IMG.INPUT_W, 3]
        img_label_shape = [None, cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.ANCHORS_NUM, cfg.IMG.PER_ANCHOR_DIM]
        img_input = tf.placeholder(dtype=tf.float32, shape=img_shape, name='img_input_placeholder')
        img_label = tf.placeholder(dtype=tf.float32, shape=img_label_shape, name='img_label_placeholder')
        img_loss_scale = tf.placeholder(dtype=tf.float32, shape=[6], name='img_loss_scale')
        trainable = tf.placeholder(dtype=tf.bool, name='training')
        img_pred = self.net(img_input, trainable)

        with tf.variable_scope('img_loss') as scope:
            img_loss = loss.img_loss(img_pred, img_label, img_loss_scale)

        return {'img_input':img_input,
                'img_label':img_label,
                'img_pred':img_pred,
                'img_loss_scale':img_loss_scale,
                'trainable':trainable,
                'yolov2_loss': img_loss[0],
                'img_obj_loss': img_loss[1],
                'img_cls_loss': img_loss[2],
                'img_bbox_loss': img_loss[3],
                }



