import os
import PIL.Image as pli
import numpy as np
import scipy.io as sio


class CountLocations:
    def __init__(self, **kwargs):
        self.data = self.get_full_path(kwargs.pop('gt_data'))
        self.numfiles = kwargs.pop('numfiles')

    def get_gt(self, index, count):
        d = sio.loadmat(
            os.path.join(self.data, 'IMG_{}_{}.mat'.format(index, count)))
        dt = d['final_gt']
        return dt

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

    def count_locations(self):
        counts = {}
        for i in range(1 ,self.numfiles + 1):
            count = 0
            for j in range(1, 10):
                gt = self.get_gt(i, j)
                _loc = gt[gt > 0]
                count += len(_loc)
            counts[i] = count 
        return counts

if __name__ == '__main__':
    print("UCF Data")
    t = CountLocations(**{
        'gt_data': 'ucf_data/gt/',
        'numfiles': 50
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)

    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))

    print()
    print("ST Data A Test")

    t = CountLocations(**{
        'gt_data': 'st_data_A_test/gt/',
        'numfiles': 182
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)


    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))


    print()
    print("ST Data A Train")

    t = CountLocations(**{
        'gt_data': 'st_data_A_train/gt/',
        'numfiles': 300
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)


    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))


    print()
    print("ST Data B Test")

    t = CountLocations(**{
        'gt_data': 'st_data_B_test/gt/',
        'numfiles': 316
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)


    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))

    print()
    print("ST Data B Train")

    t = CountLocations(**{
        'gt_data': 'st_data_B_train/gt/',
        'numfiles': 400
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)


    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))

    print()
    print("UCSD Data")   

    t = CountLocations(**{
        'gt_data': 'ucsd_data/gt/',
        'numfiles': 2000
        })
    counts = t.count_locations()
    _min = min(counts, key=counts.get)
    _max = max(counts, key=counts.get)


    print("Min: {}; Max: {}".format(counts[_min], counts[_max]))
    print("Min Image: {}; Max image: {}".format(_min, _max))


