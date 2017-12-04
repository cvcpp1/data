import os
import sys
import numpy as np 
import cv2
import random
import scipy.io as sio 
import tensorflow as tf

class DataReader():
    def __init__(self, img_path, gt_path, do_shuffle=False):
        self.img_path = img_path
        self.gt_path = gt_path
        self.filenames = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.endswith('.jpg')]
        if not len(self.filenames) > 0:
            sys.exit("No jpg files in {}".format(self.img_path))

        if do_shuffle:
            random.seed(11)
            random.shuffle(self.filenames)

        else:
            self.filenames.sort()

    def get_images(self):
        for f in self.filenames:
            img = cv2.imread(os.path.join(self.img_path, f))
            if img is None:
                sys.exit("Can't read image {}".format(f))

            if len(img.shape) > 1: #why 1, why not 2
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            X = img.reshape((1, img.shape[0], img.shape[1], 1)).astype(tf.float32)
            gt = sio.loadmat(os.path.join(self.gt_path, f.split('.')[0] + '.mat'))['final_gt']
            Y = gt.reshape((1, gt.shape[0], gt.shape[1], 1)).astype(tf.float32)

            yield (X, Y)

