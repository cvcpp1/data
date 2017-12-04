import os
import PIL.Image as pli
import numpy as np


class TestMine:
    def __init__(self, **kwargs):
        self.test_fold = self.get_full_path(kwargs.pop('test_data'))
        self.test_copy_fold = self.get_full_path(kwargs.pop('test_copy'))

        self.img_prefix = 'IMG_'
        self.gt_prefix = 'GT_IMG_'

        self.numfiles = 182

    def get_image(self, i, count, fold):
        img_filename = '{}{}_{}.jpg'.format(self.img_prefix, i + 1, count)
        img_path = os.path.join(fold, img_filename)
        img = pli.open(img_path)
        img = np.asarray(img, dtype=np.uint8)
        return img


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

    def compare(self):
        unequal = []
        for i in range(self.numfiles):
            for count in range(1, 10):
                img_mine = self.get_image(i, count, self.test_fold)
                img_copy = self.get_image(i, count, self.test_copy_fold)
                if not np.array_equal(img_mine, img_copy):
                    if img_mine.shape == img_copy.shape:
                        #print(np.subtract(img_mine, img_copy))
                        #print ((img_mine == img_copy).sum())
                        unequal.append((i + 1, count))
                    else:
                        raise Exception

        if unequal:
            print(unequal)
            print (len(unequal))
            print (9 * self.numfiles)
            d = {}
            for k, v in unequal:
                d.setdefault(k, []).append(v)

            for i in range(1, 183):
                if i not in d:
                    print ("not in d")
                    print (i)

                elif len(d[i]) != 9:
                    print ("not all images")
                    print(i)
        else:
            print( "equal")
            


if __name__ == '__main__':
    t = TestMine(**{
        'test_data': 'test_data/images/',
        'test_copy': 'test_data2/images/'

        })
    t.compare()