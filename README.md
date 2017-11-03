# Running it (Python 3)

## create a virtualenv

## activate virtualenv

## pip install -r requirements.txt

## from create_patches import CreatePatches

## See below for example code

## data should be relative to the root of the repo

# Using CreatePatches class
## Initialize with a dict of directory strings:
        img_fold: directory where  Shanghai images are stored
        gt_fold: directory where Shanghai ground truths are stored
        final_img_fold: output image folder
        final_gt_fold: output ground truth folder

## Call create_test_set() to get image and groundtruth patches

## Call plot_image_tiles or plot_dot_tiles to plot the patches

## Example Usage:

        test = CreatePatches(**{
            'img_fold': 'ST_DATA/A/test/images/',
            'gt_fold': 'ST_DATA/A/test/ground_truth/',
            'final_img_fold': 'test_data2/images/',
            'final_gt_fold': 'test_data2/gt/'
            })

        test.create_test_set()

        test.plot_image_tiles(2)
        test.plot_dot_tiles(2)

