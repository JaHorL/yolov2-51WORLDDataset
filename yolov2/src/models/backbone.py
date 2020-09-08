# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf



def yolov2_resnet22(img_input, trainable):
    with tf.variable_scope('conv_block_1') as scope: # /2x
        img_block = bl.convolutional(img_input, (3, 32), trainable, 'img_conv1')
        img_block = bl.convolutional(img_block, (3, 64), trainable, 'img_conv2', downsample=True)

    with tf.variable_scope('conv_block_2') as scope: # /4x
        for i in range(2):
            img_block = bl.resnet_blockv2(img_block, 64, 64, trainable, 'img_res%d' %i)
        img_block = bl.convolutional(img_block, (3, 128), trainable, 'img_conv1', downsample=True)

    with tf.variable_scope('conv_block_3') as scope: # /8x
        for i in range(2):
            img_block = bl.resnet_blockv2(img_block, 128, 128, trainable, 'img_res%d' %i)
        img_block = bl.convolutional(img_block, (3, 256), trainable, 'img_conv1', downsample=True)

    with tf.variable_scope('conv_block_4') as scope: # /16x
        for i in range(2):
            img_block = bl.resnet_blockv2(img_block, 256, 256, trainable, 'img_res%d' %i)
        img_block = bl.convolutional(img_block, (3, 512), trainable, 'img_conv1', downsample=True)
        route_1 = img_block

    with tf.variable_scope('conv_block_5') as scope: # /32x
        for i in range(2):
            img_block = bl.resnet_blockv2(img_block, 512, 512, trainable, 'img_res%d' %i)
        img_block = bl.convolutional(img_block, (3, 1024), trainable, 'img_conv1', downsample=True)

    return route_1, img_block


if __name__ == "__main__":
    pass