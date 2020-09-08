# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf


def yolov2_headnet(route1, img_block, img_output_z, trainable):
    with tf.variable_scope('head_block1') as scope:
        img_block = bl.convolutional(img_block, (1, 512), trainable, 'img_conv1')
        img_block = bl.convolutional(img_block, (3, 1024), trainable, 'img_conv2')
        img_block = bl.convolutional(img_block, (1, 512), trainable, 'img_conv3')
        img_block = bl.convolutional(img_block, (3, 1024), trainable, 'img_conv4')
        img_block = bl.upsample(img_block, 'upsample1')

    with tf.variable_scope('head_block2') as scope:
        img_block = tf.concat([img_block, route1], axis=-1)
        img_block = bl.convolutional(img_block, (1, 256), trainable, 'img_conv1')
        img_block = bl.convolutional(img_block, (3, 512), trainable, 'img_conv2')
        img_block = bl.convolutional(img_block, (1, 256), trainable, 'img_conv3')
        img_block = bl.convolutional(img_block, (3, 512), trainable, 'img_conv4')
        pred = bl.convolutional(img_block, (1, img_output_z), trainable, 'img_conv5')
    return pred