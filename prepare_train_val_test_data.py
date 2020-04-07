from glob import glob
import random
import os
from PIL import Image
from os.path import splitext
from os import listdir
import numpy as np
import imgaug.augmenters as iaa
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def augmentation(img, mask):
    height = int(img.shape[0] * 3 / 4)
    width = int(img.shape[1] * 3 / 4)

    rmin, rmax, cmin, cmax = bbox2(mask)

    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, : ]
    mask = mask[y:y + height, x:x + width, :]

    return img, mask



def preprocess_data(input_dir, input_mask_dir, output_dir, output_mask_dir, opts):
    sublists = opts['list']
    newW = opts['res']
    newH = opts['res']

    total_count = 0
    if sublists == None:
        sublists = ['']

    for si in range(len(sublists)):
        subject = sublists[si]
        subfiles = glob(os.path.join(input_dir, '%s*' % (subject)))
        subfiles.sort()
        subfcount = len(subfiles)
        print('%s has %d files' %(subject, subfcount))
        total_count = total_count+subfcount

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(output_mask_dir):
            os.makedirs(output_mask_dir)

        #transfer one file
        for fi in range(len(subfiles)):
            image_file = subfiles[fi]
            fbasename = os.path.basename(image_file)
            seg_file = os.path.join(input_mask_dir, fbasename)
            assert os.path.exists(seg_file)

            output_img_file = os.path.join(output_dir, fbasename)
            output_mask_file = os.path.join(output_mask_dir, fbasename)

            if os.path.exists(output_img_file) and os.path.exists(output_mask_file):
                continue

            mask = Image.open(seg_file)
            img = Image.open(image_file)

            img = np.array(img)
            mask = np.array(mask)
            [img, mask] = augmentation(img, mask)

            pil_img = Image.fromarray(img)
            pil_mask = Image.fromarray(mask)
            pil_img = pil_img.resize((newW, newH))
            pil_mask = pil_mask.resize((newW, newH))

            pil_img.save(output_img_file)
            pil_mask.save(output_mask_file)


    print('======= total has %d files' % (total_count))

    # ids = [splitext(file)[0] for file in listdir(example_dir)
    #             if not file.startswith('.')]
    # for i in range(len(ids)):
    #     idx = ids[i]
    #     idstrs = idx.split('-x-')
    #
    #
    #     mask_file = glob(os.path.join(input_dir, '%s-x-%s-x-*-x-%s-x-%s-x-%s-x%s'%(idstrs[0], idstrs[1], idstrs[2], idstrs[3], idstrs[4], idstrs[5])))
    #






### Main workflow
def main():
    random.seed(0)
    opts = {}
    opts['res'] = 512 # target resolution

    input_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/detection/ROI_images'
    output_root_dir = '/media/huoy1/48EAE4F7EAE4E264/Projects/segmentation/data/batch1_512'

    input_train_dir = os.path.join(input_root_dir, 'batch1_boarder2', 'image_iso')
    input_train_mask_dir = os.path.join(input_root_dir, 'batch1_boarder2', 'mask_iso')


    sublist = {}
    sublist['train'] = ['Case 01','Case 02','Case 06','Case 08','Case 14','Case 15','Case 17','Case 19','Case 22','Case 23','Case 24','Case 25']
    sublist['validation'] = ['Case 11', 'Case 12', 'Case 18', 'Case 20']
    sublist['test'] = ['Case 03', 'Case 05', 'Case 09', 'Case 16']


    output_train_dir = os.path.join(output_root_dir, 'imgs')
    output_train_mask_dir = os.path.join(output_root_dir, 'masks')
    opts['list'] = sublist['train']
    preprocess_data(input_train_dir, input_train_mask_dir, output_train_dir, output_train_mask_dir, opts)


    output_train_dir = os.path.join(output_root_dir, 'imgs_val')
    output_train_mask_dir = os.path.join(output_root_dir, 'masks_val')
    opts['list'] = sublist['validation']
    preprocess_data(input_train_dir, input_train_mask_dir, output_train_dir, output_train_mask_dir, opts)

    output_train_dir = os.path.join(output_root_dir, 'imgs_test')
    output_train_mask_dir = os.path.join(output_root_dir, 'masks_test')
    opts['list'] = sublist['test']
    preprocess_data(input_train_dir, input_train_mask_dir, output_train_dir, output_train_mask_dir, opts)


    #R24 data set
    input_train_dir = os.path.join(input_root_dir, 'R24_boarder2', 'image_iso')
    input_train_mask_dir = os.path.join(input_root_dir, 'R24_boarder2', 'mask_iso')
    output_train_dir = os.path.join(output_root_dir, 'imgs_external_test')
    output_train_mask_dir = os.path.join(output_root_dir, 'masks_external_test')
    opts['list'] = None
    preprocess_data(input_train_dir, input_train_mask_dir, output_train_dir, output_train_mask_dir, opts)



if __name__ == '__main__':
    main()