import os
import io
import time
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import utils
from data import dataset
from data import postprocess
from utils import vis_tools
from models import yolov2_network
from utils import math
from utils import timer
import json
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession




class predicter(object):

    def __init__(self):
        self.initial_weight      = cfg.EVAL.WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLOv2.MOVING_AVE_DECAY
        self.eval_logdir        = "./data/logs/eval"
        self.evalset             = dataset.Dataset('test')
        self.output_dir          = cfg.EVAL.OUTPUT_PRED_PATH
        self.img_anchors         = loader.load_anchors(cfg.IMG.ANCHORS)

        with tf.name_scope('model'):
            self.model               = yolov2_network.YOLOv2Network()
            self.net                 = self.model.load()
            self.img_pred            = self.net['img_pred']

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = InteractiveSession(config=config)
        self.saver = tf.train.Saver()#ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.initial_weight)
        self.timer = timer.Timer()


    def predict(self):
        img_imwrite_path = os.path.join(self.output_dir, "img_imshow_result/")
        img_result_path  = os.path.join(self.output_dir, "img_result/")
        if os.path.exists(img_imwrite_path):
            shutil.rmtree(img_imwrite_path)
        os.mkdir(img_imwrite_path)
        if os.path.exists(img_result_path):
            shutil.rmtree(img_result_path)
        os.mkdir(img_result_path)
        dumped_json = {"tps":[], "fps":[], "fns":[]}
        for step in range(len(self.evalset)):
        # for step in range(10):
            print(step, "/", len(self.evalset))
            eval_result = self.evalset.load()
            # print("load time: ", self.timer.time_diff_per_n_loops())
            img_pred = self.sess.run(self.img_pred,    
                                     feed_dict={self.net["img_input"]: eval_result[0],
                                     self.net["trainable"]: False})[0][0]
            # print("inference time: ", self.timer.time_diff_per_n_loops())
            img_bboxes = postprocess.parse_img_predmap(img_pred, self.img_anchors)
            img_bboxes = postprocess.img_nms(img_bboxes, cfg.IMG.IOU_THRESHOLDS)
            # print("postprocess time: ", self.timer.time_diff_per_n_loops())


if __name__ == "__main__":
    predicter = predicter()
    predicter.predict()
