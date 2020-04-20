import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.visual_img import show_src_img_label

import torch.backends.cudnn as cudnn
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True
     #cudnn.benchmark = False
     #cudnn.enabled = False

setup_seed(5125)            #原因是标注数据出了问题，而之所以不好debug的原因可能是因为改过程为gpu运算，不容易追踪错误的来源。
# setup_seed(612)


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


# win10 coco2017
class ListDataset_win_cc2017(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        # there are seven rows before the first file name row.
        for i in range(7):
            self.img_files.pop(0)
        
        # the last two rows is valid
        self.img_files.pop(-1)
        self.img_files.pop(-1)

        # absolute path of images
        if "train" in list_path:
            fore_path = "\\".join(list_path.split("\\")[:-1]) + "\\train2017\\train2017_images\\"
        elif "val" in list_path:
            fore_path = "\\".join(list_path.split("\\")[:-1]) + "\\val2017\\val2017_images\\"

        # leave the last 17 and remove "\n"
        for i in range(len(self.img_files)):
            self.img_files[i] = fore_path + self.img_files[i][-17:][:-1]
        # print("self.img_files: ", self.img_files[-1])

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        # print("self.label_files: ", self.label_files[-1])
        

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # print("img_path: ", img_path)  # to freeze the random seed

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # print("load img.shape: ", img.shape)     # img.shape:  torch.Size([3, 375, 500])

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        # 当正方化标签时，h_factor 和 w_factor 分别是图片的高度和宽度
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        # print("pad img.shape: ", img.shape)
        # 填充后，padded_h == padded_w，为正方形
        _, padded_h, padded_w = img.shape
        # print(padded_h, padded_w)   # (640, 640)

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        # print("label_path: ", label_path)       # label_path:  E:\dataset_work\coco\train2017\train2017_labels\000000032270.txt

        targets = None
        if os.path.exists(label_path):
            # print("$"*50)
            np_boxes = np.loadtxt(label_path).reshape(-1, 5)
            np_boxes = np.where(np_boxes > 79, 79, np_boxes)            # 但是此处实际上是进行了一个错误处理，因为这个标注是错的，应该pass掉
            boxes = torch.from_numpy(np_boxes)
            # print("load boxes[0]: ", boxes[0].numpy())
            # change the label into float
            # (boxes[:, 1], boxes[:, 2]) is top left, boxes[:, 3] is height, boxes[:, 4] is width 
            boxes[:, 1] /= w_factor
            boxes[:, 3] /= w_factor
            boxes[:, 2] /= h_factor
            boxes[:, 4] /= h_factor
            # print("change float boxes[0]: ", boxes[0].numpy())
            # print("w_factor, h_factor", w_factor, h_factor)
            # Extract coordinates for unpadded + unscaled image

            # x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            # y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            # x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            # y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            x1 = w_factor * (boxes[:, 1])
            y1 = h_factor * (boxes[:, 2])
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3])
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4])

            # print("Extract: ", x1[0].numpy(), y1[0].numpy(), x2[0].numpy(), y2[0].numpy())
            
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # print("pad: ", pad)
            # print("after pad: ", x1[0].numpy(), y1[0].numpy(), x2[0].numpy(), y2[0].numpy())
            # Returns (x, y, w, h)
            # ******Returns (x, y, w, h)********
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            # ***********************************

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
            # print("pad: ", pad)
            # print("target: ",targets[0]) 

        # # 可视化
        # show_src_img_label(img_path, label_path)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        
        # print("img.shape end: ", img.shape)
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # print("collate_fn function input: imgs ", imgs[0].shape)
        # print("collate_fn function input: targets ", targets)
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            # print("here is a multiscale training, batch_count: ", self.batch_count)
            # print("self.min_size, self.max_size", self.min_size, self.max_size + 1)
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            # print("new size: ", self.img_size)
            # print("self.img_size", self.img_size)
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1

        # print("img_path: ", paths)
        # print("img: ", imgs.shape)
        # print("target: ", targets.shape)
        # print("%"*50)

        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
