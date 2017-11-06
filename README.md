# Running it (Python 3)

- create a virtualenv

- activate virtualenv

- pip install -r requirements.txt

- from create_patches import CreatePatches, CreatePatchesUCSD

- See below for example code

- data should be relative to the root of the repo

# Using CreatePatches class
- Initialize with a dict of directory strings and the correct config for the dataset:

        img_fold: directory where images are stored
        gt_fold: directory where ground truths are stored
        final_img_fold: output image folder
        final_gt_fold: output ground truth folder

- Call create_test_set() to get image and groundtruth patches
- Call plot_image_tiles or plot_dot_tiles to plot the patches

# Example Usage:

            inputs = {
                'img_fold': 'ST_DATA/A/test/images/',
                'gt_fold': 'ST_DATA/A/test/ground_truth/',
                'final_img_fold': 'st_data_A_test/images/',
                'final_gt_fold': 'st_data_A_test/gt/'

            }
            inputs.update(**ST_DATA_CONFIG)
            test = CreatePatches(**inputs)
            test.create_test_set()
            test.plot_image_tiles(2)
            test.plot_dot_tiles(2)

            inputs = {
                'img_fold': 'ST_DATA/A/train/images/',
                'gt_fold': 'ST_DATA/A/train/ground_truth/',
                'final_img_fold': 'st_data_A_train/images/',
                'final_gt_fold': 'st_data_A_train/gt/'

            }
            inputs.update(**ST_DATA_CONFIG)
            test = CreatePatches(**inputs)
            test.create_test_set()
            test.plot_image_tiles(2)
            test.plot_dot_tiles(2)


            inputs = {
                'img_fold': 'ST_DATA/B/test_data/images/',
                'gt_fold': 'ST_DATA/B/test_data/ground_truth/',
                'final_img_fold': 'st_data_B_test/images/',
                'final_gt_fold': 'st_data_B_test/gt/'

            }
            inputs.update(**ST_DATA_CONFIG)
            test = CreatePatches(**inputs)
            test.create_test_set()
            test.plot_image_tiles(2)
            test.plot_dot_tiles(2)

            inputs = {
                'img_fold': 'ST_DATA/B/train_data/images/',
                'gt_fold': 'ST_DATA/B/train_data/ground_truth/',
                'final_img_fold': 'st_data_B_train/images/',
                'final_gt_fold': 'st_data_B_train/gt/'

            }
            inputs.update(**ST_DATA_CONFIG)
            test = CreatePatches(**inputs)
            test.create_test_set()
            test.plot_image_tiles(2)
            test.plot_dot_tiles(2)    


            inputs = {
                'img_fold': 'UCF_CC_50/',
                'gt_fold': 'UCF_CC_50/',
                'final_img_fold': 'ucf_data/images/',
                'final_gt_fold': 'ucf_data/gt/',

            }
            inputs.update(**UCF_DATA_CONFIG)

            test = CreatePatches(**inputs)
            test.create_test_set()
            test.plot_image_tiles(1)
            test.plot_dot_tiles(1)



            inputs = {
                'img_fold': 'ucsdpeds/vidf/',
                'gt_fold':  'gt_1_33/',
                'final_img_fold': 'ucsd_data/images/',
                'final_gt_fold': 'ucsd_data/gt/'
            }
            inputs.update(**UCSD_DATA_CONFIG)
            test = CreatePatchesUCSD(**inputs)
            test.create_test_set()
            test.plot_image_tiles(40)
            test.plot_dot_tiles(40)

