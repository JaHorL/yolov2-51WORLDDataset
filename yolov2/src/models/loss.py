import tensorflow as tf 
from config.config import cfg


def smooth_l1(deltas, targets, sigma=2.0):
	'''
	ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
	SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
					|x| - 0.5 / sigma^2,    otherwise
	'''
	sigma2 = sigma * sigma
	diffs  =  tf.subtract(deltas, targets)
	l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

	l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
	l1_option2 = tf.abs(diffs) - 0.5 / sigma2
	l1_add = tf.multiply(l1_option1, l1_signs) + \
						tf.multiply(l1_option2, 1-l1_signs)
	l1 = l1_add

	return l1



def img_loss(pred, label, img_loss_scale):
	
	cls_num = cfg.YOLOv2.CLASSES_NUM
	epsilon = cfg.YOLOv2.EPSILON
	mask = tf.cast(label[...,0] ,tf.bool)
	batch_size  = tf.cast(tf.shape(pred),tf.float32)[0]
	shape = [-1, cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, cfg.IMG.ANCHORS_NUM, 
			cfg.IMG.PER_ANCHOR_DIM]
	pred = tf.reshape(pred, shape)
	with tf.name_scope('mask'):
		masked_label = tf.boolean_mask(label, mask)
		masked_pred = tf.boolean_mask(pred, mask)
		masked_neg_pred = tf.boolean_mask(pred, tf.logical_not(mask))
 
	with tf.name_scope('pred'):
		pred_conf = tf.sigmoid(masked_pred[..., 0])
		pred_neg_conf = tf.sigmoid(masked_neg_pred[..., 0])
		pred_c = tf.sigmoid(masked_pred[..., 1:cls_num+1])
		pred_xy = tf.sigmoid(masked_pred[..., 1+cls_num:3+cls_num]) 
		pred_hw = tf.exp(masked_pred[..., 3+cls_num:]) 
	
	with tf.name_scope('label'):
		label_c = tf.sigmoid(masked_label[..., 1:cls_num+1])
		label_xy = masked_label[..., 1+cls_num:3+cls_num]
		label_hw= masked_label[..., 3+cls_num:]

	with tf.name_scope('loss'):
		positive_obj_loss = tf.reduce_sum(-tf.log(pred_conf + epsilon)) 
		negative_obj_loss = tf.reduce_sum(-tf.log(1 - pred_neg_conf + epsilon))
		positive_cls_loss = tf.reduce_sum(-tf.log(pred_c + epsilon) * label_c) 
		negative_cls_loss = tf.reduce_sum(-tf.log(1- pred_c + epsilon) * (1 - label_c))
		xy_loss	= tf.reduce_sum(smooth_l1(pred_xy, label_xy))
		hw_loss = tf.reduce_sum(smooth_l1(pred_hw, label_hw))
		
		objness_loss = (positive_obj_loss * 5 + negative_obj_loss * 0.1) / batch_size 
		cls_loss = (positive_cls_loss + negative_cls_loss) / batch_size
		bbox_loss = (xy_loss + hw_loss) / batch_size
		total_loss = bbox_loss * 5 + objness_loss + cls_loss	
	return total_loss, objness_loss, cls_loss, bbox_loss

