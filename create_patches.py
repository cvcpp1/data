import scipy.io as sio
import os
import numpy as np
import PIL.Image as pli
import math
import multiprocessing as mlt 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
    

class CreatePatches:
    ''' 
    Initialize with a dict of directory strings:
        img_fold: directory where  Shanghai images are stored
        gt_fold: directory where Shanghai ground truths are stored
        final_img_fold: output image folder
        final_gt_fold: output ground truth folder

    Call create_test_set() to get image and groundtruth patches

    Call plot_image_tiles or plot_dot_tiles to plot the patches

    Example Usage:

        test = CreatePatches(**{
            'img_fold': 'ST_DATA/A/test/images/',
            'gt_fold': 'ST_DATA/A/test/ground_truth/',
            'final_img_fold': 'test_data2/images/',
            'final_gt_fold': 'test_data2/gt/'
            })

        test.create_test_set()

        test.plot_image_tiles(2)
        test.plot_dot_tiles(2)

    '''


    def __init__(self, **kwargs):
        self.img_fold = self.get_full_path(kwargs.pop('img_fold'))
        self.gt_fold = self.get_full_path(kwargs.pop('gt_fold'))
        self.final_img_fold = self.get_full_path(kwargs.pop('final_img_fold'), True)
        self.final_gt_fold = self.get_full_path(kwargs.pop('final_gt_fold'), True)
        self.img_prefix = 'IMG_'
        self.gt_prefix = 'GT_IMG_'
        self.numfiles = len([f for f in os.listdir(self.img_fold) if f.endswith('.jpg') and os.path.isfile(os.path.join(self.img_fold, f))])


    def get_full_path(self, rel_path, makedir=False):
        directory = os.path.join(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            ),
            rel_path
        )
        if makedir:
            if not os.path.exists(directory):
                os.makedirs(directory)

        return directory

    def get_image(self, i):
        img_filename = '{}{}.jpg'.format(self.img_prefix, i + 1)
        img_path = os.path.join(self.img_fold, img_filename)
        img = pli.open(img_path)
        img = np.asarray(img, dtype=np.uint8)
        return img

    def save_image(self, img, i, count):
        name = '{}{}_{}.jpg'.format(self.img_prefix, i + 1, count)
        img = np.uint8(img)
        img = pli.fromarray(img).save(os.path.join(self.final_img_fold, name))

    def get_ground_truth(self, i):
        gt_filename = '{}{}.mat'.format(self.gt_prefix, i + 1)
        gt_path = os.path.join(self.gt_fold, gt_filename)
        gt = sio.loadmat(gt_path)
        image_info = gt['image_info']
        value = image_info[0,0]
        assert len(value['location']) == 1
        for i in value['location']:
            assert len(i) == 1
            for j in i:
                return j     

    def save_gt(self, gt, i, count):
         name = '{}{}_{}.mat'.format(self.img_prefix, i + 1, count)
         sio.savemat(os.path.join(self.final_gt_fold, name), {'final_gt': gt})

    def create_dotmaps(self, gt, img_h, img_w):
        d_map = np.zeros((int(img_h), int(img_w)))

        gt = gt[gt[:, 0] < img_w, :]
        gt = gt[gt[:, 1] < img_h, :]

        for i in range(gt.shape[0]):
            x = int(max(1, math.floor(gt[i, 0]))) - 1
            y = int(max(1, math.floor(gt[i, 1]))) - 1
            d_map[y, x] = 1.0
        return d_map

    class DimensionException(Exception):
        pass

    def check_dim(self, img):
        if img.ndim != 3:
            if img.ndim == 2:
                img = np.stack((img,)*3, axis=2)
            else:
                raise DimensionException("Image has incorrect dimensions. {}".format(img.shape))
        return img

    def _create_test_set(self, i):
        print(i + 1)
        img = self.get_image(i)
        # moved this out of loop because 3rd dim indexing doesn't work in numpy unless already that shape
        img = self.check_dim(img)
        gt = self.get_ground_truth(i)

        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

        d_map = self.create_dotmaps(gt / 4.0, d_map_h, d_map_w)

        p_h = int(math.floor(float(img.shape[0]) / 3.0))
        p_w = int(math.floor(float(img.shape[1]) / 3.0))
        d_map_ph = int(math.floor(math.floor(p_h / 2.0) / 2.0))
        d_map_pw = int(math.floor(math.floor(p_w / 2.0) / 2.0))
        
        py = 0
        py2 = 0
        count = 1

        for j in range(3):
            px = 0
            px2 = 0
            for k in range(3):
                final_image = img[py:py + p_h, px: px + p_w, :]
                final_gt = d_map[py2: py2 + d_map_ph, px2: px2 + d_map_pw]                 
                px = px + p_w 
                px2 = px2 + d_map_pw
                self.save_image(final_image, i, count)
                self.save_gt(final_gt, i, count)
                count += 1
            py = py + p_h
            py2 = py2 + d_map_ph  


    def create_test_set(self):
        p = mlt.Pool(mlt.cpu_count())
        p.map(self._create_test_set, range(self.numfiles))

        #for i in range(self.numfiles):
            #self._create_test_set(i)

    def plot_image_tiles(self, index):
        fig = plt.figure()
        count = 1
        for i in range(3):
            for j in range(3):
                a=fig.add_subplot(3,3,count)
                a.set_xticks([])
                a.set_yticks([])
                b = mpimg.imread(
                    os.path.join(self.final_img_fold, 'IMG_{}_{}.jpg'.format(index, count)))
                imgplot = plt.imshow(b)
                count += 1
        plt.subplots_adjust(left=None, bottom=.18, right=None, top=None, wspace=.01, hspace=.001)
        plt.show()

    def plot_dot_tiles(self, index):
        fig = plt.figure()
        count = 1
        for i in range(3):
            for j in range(3):
                a=fig.add_subplot(3,3,count)
                a.set_xticks([])
                a.set_yticks([])
                d = sio.loadmat(
                    os.path.join(self.final_gt_fold, 'IMG_{}_{}.mat'.format(index, count)))
                dt = d['final_gt']
                imgplot = plt.imshow(dt, cmap='gray')
                count += 1
        plt.subplots_adjust(left=None, bottom=.18, right=None, top=None, wspace=.01, hspace=.001)
        plt.show()
        












