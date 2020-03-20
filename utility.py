import os
import math
import shutil
import zipfile
import tarfile
import numpy as np
import cv2 as cv
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import random
import argparse
import logging


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(raw_data_path, im_name, bg_name, fcount, bcount, mode):
    fg_path = os.path.join(raw_data_path, '{}/fg/'.format(mode))
    a_path = os.path.join(raw_data_path, '{}/mask/'.format(mode))
    bg_path = os.path.join(raw_data_path, '{}/bg/'.format(mode))
    out_path = os.path.join(raw_data_path, '{}/merged/'.format(mode))
    
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out = composite4(im, bg, a, w, h)
    if mode == "train":
        filename = out_path + str(fcount) + '_' + str(bcount) + '.png'
    elif mode == "test":
        filename = out_path  + bg_name.split('.')[0]+'!'+im_name.split('.')[0]+'!'+ str(fcount) + '!' + str(bcount) + '.png'
    cv.imwrite(filename, out)


def process_one_fg(params):
    fcount = params[0]
    raw_data_path = params[1]
    num_bgs = params[2]
    fg_files = params[3]
    mode = params[4]

    folder = 'Training_set' if mode == 'train' else 'Test_set'
    txt_name = 'training_bg_names' if mode == 'train' else 'test_bg_names'
    with open(os.path.join(raw_data_path, 'Combined_Dataset', folder, '{}.txt'.format(txt_name))) as f:
        bg_files = f.read().splitlines()

    im_name = fg_files[fcount]
    bcount = fcount * num_bgs

    for _ in range(num_bgs):
        bg_name = bg_files[bcount]
        process(raw_data_path, im_name, bg_name, fcount, bcount, mode)
        bcount += 1


def do_composite(raw_data_path, num_bgs, mode):
    folder = 'Training_set' if mode == 'train' else 'Test_set'
    txt_name = 'training_fg_names' if mode == 'train' else 'test_fg_names'
    with open(os.path.join(raw_data_path, 'Combined_Dataset', folder, '{}.txt').format(txt_name)) as f:
        fg_files = f.read().splitlines()

    with Pool(processes=16) as p:
        max_ = len(fg_files)
        params = []
        for i in range(max_):
            params.append([i, raw_data_path, num_bgs, fg_files, mode])
        with tqdm(total=max_) as pbar:
            for _, _ in tqdm(enumerate(p.imap_unordered(process_one_fg, params))):
                pbar.update()


def composite_dataset(raw_data_path, logger):
    # Path to provided foreground images
    fg_path = os.path.join(raw_data_path, 'train/fg/')
    # Path to provided alpha mattes
    a_path = os.path.join(raw_data_path, 'train/mask/')
    # Path to background images (MSCOCO)
    bg_path = os.path.join(raw_data_path, 'train/bg/')
    # Path to folder where you want the composited images to go
    out_path = os.path.join(raw_data_path, 'train/merged/')

    train_folder = os.path.join(raw_data_path, 'Combined_Dataset/Training_set/')

    # Extract Adobe dataset
    if not os.path.exists(os.path.join(raw_data_path, 'Combined_Dataset/')):
        zip_file = os.path.join(raw_data_path, 'Adobe_Deep_Matting_Dataset.zip')
        logger.info('Extracting Adobe_Deep_Matting_Dataset.zip')
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(raw_data_path)
        zip_ref.close()
    
    # Extract train2014
    if not os.path.exists(os.path.join(raw_data_path, 'train2014/')):
        zip_file = os.path.join(raw_data_path, 'train2014.zip')
        logger.info('Extracting train2014.zip')
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(raw_data_path)
        zip_ref.close()
    
    # Move training background images into designated folder
    if not os.path.exists(bg_path):
        logger.info('Moving training background images into designated folder')
        with open(os.path.join(train_folder, 'training_bg_names.txt')) as f:
            training_bg_names = f.read().splitlines()
        os.makedirs(bg_path)
        for bg_name in training_bg_names:
            src_path = os.path.join(raw_data_path, 'train2014', bg_name)
            dest_path = os.path.join(bg_path, bg_name)
            shutil.move(src_path, dest_path)
    
    # Move training foreground images into designated folder
    if not os.path.exists(fg_path):
        logger.info('Moving training foreground images into designated folder')
        os.makedirs(fg_path)
        for old_folder in [train_folder + 'Adobe-licensed images/fg', train_folder + 'Other/fg']:
            fg_files = os.listdir(old_folder)
            for fg_file in fg_files:
                src_path = os.path.join(old_folder, fg_file)
                dest_path = os.path.join(fg_path, fg_file)
                shutil.move(src_path, dest_path)

    # Move training alpha images into designated folder
    if not os.path.exists(a_path):
        logger.info('Moving training alpha images into designated folder')
        os.makedirs(a_path)
        for old_folder in [train_folder + 'Adobe-licensed images/alpha', train_folder + 'Other/alpha']:
            a_files = os.listdir(old_folder)
            for a_file in a_files:
                src_path = os.path.join(old_folder, a_file)
                dest_path = os.path.join(a_path, a_file)
                shutil.move(src_path, dest_path)

    # Make the folder for composited training images
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        logger.info('Compositing training images')
        do_composite(raw_data_path, 100, "train")
        logger.info('Training images composited')

    # Path to provided foreground images
    fg_test_path = os.path.join(raw_data_path, 'test/fg/')
    # Path to provided alpha mattes
    a_test_path = os.path.join(raw_data_path, 'test/mask/')
    # Path to background images (PASCAL VOC)
    bg_test_path = os.path.join(raw_data_path, 'test/bg/')
    # Path to folder where you want the composited images to go
    out_test_path = os.path.join(raw_data_path, 'test/merged/')

    test_folder = os.path.join(raw_data_path, 'Combined_Dataset/Test_set/')

    if not os.path.exists(os.path.join(raw_data_path, 'VOCdevkit')):
        # Extract VOCtrainval
        tar_file = os.path.join(raw_data_path, 'VOCtrainval_14-Jul-2008.tar')
        logger.info('Extracting VOCtrainval_14-Jul-2008.tar')
        tar = tarfile.open(tar_file)
        tar.extractall(raw_data_path)
        tar.close()
        # Extract VOCtest
        tar_file = os.path.join(raw_data_path, 'VOC2008test.tar')
        logger.info('Extracting VOC2008test.tar')
        tar = tarfile.open(tar_file)
        tar.extractall(raw_data_path)
        tar.close()

    # Move testing background images into designaed folder
    if not os.path.exists(bg_test_path):
        logger.info('Moving testing background images into designated folder')
        os.makedirs(bg_test_path)
        with open(os.path.join(test_folder, 'test_bg_names.txt')) as f:
            test_bg_names = f.read().splitlines()

        for bg_name in test_bg_names:
            _ = bg_name.split('_')
            src_path = os.path.join(raw_data_path, 'VOCdevkit/VOC2008/JPEGImages', bg_name)
            dest_path = os.path.join(bg_test_path, bg_name)
            shutil.move(src_path, dest_path)
    
    # Move testing foreground images into designated folder
    if not os.path.exists(fg_test_path):
        logger.info('Moving testing foreground images into designated folder')
        os.makedirs(fg_test_path)
        for old_folder in [test_folder + 'Adobe-licensed images/fg']:
            fg_files = os.listdir(old_folder)
            for fg_file in fg_files:
                src_path = os.path.join(old_folder, fg_file)
                dest_path = os.path.join(fg_test_path, fg_file)
                shutil.move(src_path, dest_path)

    # Move testing alpha images into desigated folder
    if not os.path.exists(a_test_path):
        logger.info('Moving testing alpha images into designated folder')
        os.makedirs(a_test_path)
        for old_folder in [test_folder + 'Adobe-licensed images/alpha']:
            a_files = os.listdir(old_folder)
            for a_file in a_files:
                src_path = os.path.join(old_folder, a_file)
                dest_path = os.path.join(a_test_path, a_file)
                shutil.move(src_path, dest_path)

    # Make the folder for composited testing images
    if not os.path.exists(out_test_path):
        os.makedirs(out_test_path)
        logger.info('Compositing testing images')
        do_composite(raw_data_path, 20, "test")
        logger.info('Testing images composited')


def gen_train_valid_names(logger):
    logger.info("Start generating train/valid name files")
    num_fgs = 431
    # num_bgs = 43100
    num_bgs_per_fg = 100
    num_valid = 8620

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for _ in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('dataset/valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('dataset/train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))
    logger.info("Generated train/valid name files")


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='set arguments')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "prep"], help="set the program to \'train\', \'test\', or \'prep\'")
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=str, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=str, required=True, help="crop width size of input image")
    parser.add_argument('--alphaDir', type=str, required=True, help="directory of alpha")
    parser.add_argument('--fgDir', type=str, required=True, help="directory of fg")
    parser.add_argument('--bgDir', type=str, required=True, help="directory of bg")
    parser.add_argument('--imgDir', type=str, default="", help="directory of img")
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=10, help='epoch of learning decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--gpu', type=str, default="0", help="choose gpus")
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    parser.add_argument('--saveDir', type=str, help="checkpoint that model save to")
    parser.add_argument('--printFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--ckptSaveFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--wl_weight', type=float, default=0.5, help="alpha loss weight")
    parser.add_argument('--testFreq', type=int, default=-1, help="test frequency")
    parser.add_argument('--testImgDir', type=str, default='', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='', help="test result save to")
    parser.add_argument('--crop_or_resize', type=str, default="whole", choices=["resize", "crop", "whole"], help="how manipulate image before test")
    parser.add_argument('--max_size', type=int, default=1312, help="max size of test image")
    parser.add_argument('--log', action="store_true", default=False, help="whether store log to log.txt")
    parser.add_argument('--debug', action="store_true", default=False, help="whether print verbose debug info")
    parser.add_argument('--raw_data_path', type=str, default="/data/datasets/im/AdaMatting/", help="dir where datasets are stored")
    args = parser.parse_args()
    return args


def get_logger(flag):
    logger = logging.getLogger("AdaMatting")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s %(lineno)d: %(levelname)s - %(message)s")

    # log file stream
    if (flag):
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger
