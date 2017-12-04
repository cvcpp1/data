import numpy as np
import scipy

def create_density(gt, img_h, img_w):
	K = 7
	m = img_h
	n = img_w

	d_map = np.zeros((m,n))

    gt = gt[gt[:, 0] < img_w, :]
    gt = gt[gt[:, 1] < img_h, :]

    for i in range(gt.shape[0]):
    	pass # need to figure out what matlab knnsearch is returning