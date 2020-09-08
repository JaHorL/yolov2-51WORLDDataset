import os
import cv2
import time
import ctypes
import threading
import numpy as np
from config.config import cfg
from utils import utils
from utils import vis_tools
from utils import timer
from data import labels
from loader import simone_loader
from loader.loader_config import loader_cfg

class Dataset(object):
    def __init__(self, process_type):
        if process_type == 'train':
            self.anno_path     = cfg.YOLOv2.TRAIN_DATA
            self.batch_size    = cfg.TRAIN.BATCH_SIZE
            self.is_training   = True
            self.data_type     = loader_cfg.TRAINING_LOADER_FLAGS
        else:
            self.anno_path     = cfg.YOLOv2.TEST_DATA
            self.batch_size    = cfg.EVAL.BATCH_SIZE
            self.is_training   = False
        self.dataset_loader    = simone_loader.SimoneDatasetLoader(loader_cfg.DATASET_DIR, loader_cfg.TRAINING_LOADER_FLAGS, True)
        self.num_samples       = self.dataset_loader.get_total_num()
        self.num_batchs        = int(np.ceil(self.num_samples / self.batch_size)-2)
        self.batch_count       = 0
        self.is_use_thread     = cfg.YOLOv2.IS_USE_THREAD
        self.img_anchors       = self.load_anchors(cfg.IMG.ANCHORS)

        self.loader_need_exit = 0
        self.timer = timer.Timer()
        self.per_step_ano = []
        if self.is_use_thread:
            self.prepr_data = []
            self.max_cache_size = 10
            self.lodaer_processing = threading.Thread(target=self.loader)
            self.lodaer_processing.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit = True
        print('set loader_need_exit True')
        self.lodaer_processing.join()
        print('exit lodaer_processing')

    def __len__(self):
        return self.num_batchs

    def preprocess_data(self):
        batch_img = np.zeros((self.batch_size, cfg.IMG.INPUT_H, cfg.IMG.INPUT_W, 3), dtype=np.float32)
        batch_img_label = np.zeros((self.batch_size, cfg.IMG.OUTPUT_H, cfg.IMG.OUTPUT_W, 
                        cfg.IMG.ANCHORS_NUM, cfg.IMG.PER_ANCHOR_DIM), dtype=np.float32)
        batch_frame_id = []
        for i in range(self.batch_size):
            data_info = self.dataset_loader.next()
            batch_img_label[i, ...] = labels.create_img_label(data_info['image_annos'], self.img_anchors).astype(np.float32)
            batch_img[i, ...] = cv2.resize(data_info['image'], (cfg.IMG.INPUT_W, cfg.IMG.INPUT_H), cv2.INTER_CUBIC).astype(np.float32)
        self.batch_count += 1
        return [batch_img, batch_img_label, batch_frame_id]

    def load_anchors(self, anchors_path):
        with open(anchors_path) as f:
            anchors = f.readlines()
        new_anchors = np.zeros([len(anchors), len(anchors[0].split())], dtype=np.float32)
        for i in range(len(anchors)):
            new_anchors[i] = np.array(anchors[i].split(), dtype=np.float32)
        return new_anchors

    def loader(self):
        while(not self.loader_need_exit):
            if len(self.prepr_data) < self.max_cache_size: 
                self.prepr_data.append(self.preprocess_data())
            else:
                time.sleep(0.01)
                self.loader_need_exit = False

    def load(self):
        if self.is_use_thread:
            while len(self.prepr_data) == 0:
                time.sleep(0.01)
            data_ori = self.prepr_data.pop()
        else:
            data_ori = self.preprocess_data()
        if self.batch_count >= self.num_batchs:
            self.batch_count = 0
        return data_ori
                                                   

if __name__ == "__main__":
	pass
