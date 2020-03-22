import os
import math
import random
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


data_transforms = {
    # values from ImageNet
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class AdaMattingDataset(Dataset):

    def __init__(self, raw_data_path, mode):
        self.crop_size = 320
        self.unknown_code = 128
        self.mode = mode
        self.raw_data_path = raw_data_path

        self.fg_path = os.path.join(self.raw_data_path, "train/fg/")
        self.bg_path = os.path.join(self.raw_data_path, "train/bg/")
        self.a_path = os.path.join(self.raw_data_path, "train/mask/")

        self.transformer = data_transforms[self.mode]

        with open(os.path.join(self.raw_data_path, "Combined_Dataset/Training_set/training_fg_names.txt")) as f:
            self.fg_files = f.read().splitlines()
        with open(os.path.join(self.raw_data_path, "Combined_Dataset/Training_set/training_bg_names.txt")) as f:
            self.bg_files = f.read().splitlines()

        filename = "dataset/{}_names.txt".format(self.mode)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()
    

    def __len__(self):
        return len(self.names)


    def composite4(self, fg, bg, a, w, h):
        fg = np.array(fg, np.float32)
        bg_h, bg_w = bg.shape[:2]
        x = 0
        if bg_w > w:
            x = np.random.randint(0, bg_w - w)
        y = 0
        if bg_h > h:
            y = np.random.randint(0, bg_h - h)
        bg = np.array(bg[y:y + h, x:x + w], np.float32)
        alpha = np.zeros((h, w, 1), np.float32)
        alpha[:, :, 0] = a / 255.
        im = alpha * fg + (1 - alpha) * bg
        im = im.astype(np.uint8)
        return im, a, fg, bg


    def process(self, im_name, bg_name):
        im = cv.imread(self.fg_path + im_name)
        a = cv.imread(self.a_path + im_name, 0)
        h, w = im.shape[:2]
        bg = cv.imread(self.bg_path + bg_name)
        bh, bw = bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

        return self.composite4(im, bg, a, w, h)
    

    def __getitem__(self, index):
        name = self.names[index]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = self.fg_files[fcount]
        bg_name = self.bg_files[bcount]
        img, alpha, _, _ = self.process(im_name, bg_name)

        different_sizes = [(320, 320), (800, 800)]
        crop_size = random.choice(different_sizes)

        trimap = self.gen_trimap(alpha)
        x, y = self.random_choice(trimap, crop_size)
        img = self.safe_crop(img, x, y, crop_size)
        alpha = self.safe_crop(alpha, x, y, crop_size)

        trimap = self.gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        x = torch.zeros((4, self.crop_size, self.crop_size), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[0:3, :, :] = img
        x[3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        y = np.empty((2, self.crop_size, self.crop_size), dtype=np.float32)
        y[0, :, :] = alpha / 255.
        # mask = np.equal(trimap, 128).astype(np.float32)
        """
        pred_trimap_argmax
        0: background
        1: unknown
        2: foreground
        """
        mask = np.zeros(alpha.shape)
        mask.fill(1)
        mask[alpha <= 0] = 0
        mask[alpha >= 255] = 2
        y[1, :, :] = mask
        return x, y


    def gen_trimap(self, alpha):
        k_size = random.choice(range(1, 5))
        iterations = np.random.randint(1, 20)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv.dilate(alpha, kernel, iterations)
        eroded = cv.erode(alpha, kernel, iterations)
        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        return trimap


    # Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
    def random_choice(self, trimap, crop_size=(320, 320)):
        crop_height, crop_width = crop_size
        y_indices, x_indices = np.where(trimap == self.unknown_code)
        num_unknowns = len(y_indices)
        x, y = 0, 0
        if num_unknowns > 0:
            ix = np.random.choice(range(num_unknowns))
            center_x = x_indices[ix]
            center_y = y_indices[ix]
            x = max(0, center_x - int(crop_width / 2))
            y = max(0, center_y - int(crop_height / 2))
        return x, y


    def safe_crop(self, mat, x, y, crop_size):
        crop_height, crop_width = crop_size
        if len(mat.shape) == 2:
            ret = np.zeros((crop_height, crop_width), np.uint8)
        else:
            ret = np.zeros((crop_height, crop_width, 3), np.uint8)
        crop = mat[y:y + crop_height, x:x + crop_width]
        h, w = crop.shape[:2]
        ret[0:h, 0:w] = crop
        if crop_size != (self.crop_size, self.crop_size):
            ret = cv.resize(ret, dsize=(self.crop_size, self.crop_size), interpolation=cv.INTER_NEAREST)
        return ret