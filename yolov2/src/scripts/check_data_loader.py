import sys
sys.path.append("../")
import os
import cv2
import shutil
from data import dataset
from utils import utils
from utils import vis_tools
from tqdm import tqdm
from config.config import cfg
from data import postprocess
# from loader.loader_config import loader
import numpy as np
from utils import math


trainset = dataset.Dataset('train')
save_dir = '/home/jhli/Work/doc/simone_img_analysis/samples/'       
src_dir = '/home/jhli/Data/image/simone/'

for i in range(10):
    for j in range(len(trainset)):
        data = trainset.load()