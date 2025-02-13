import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio.v2 as imageio
from . import transforms
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()


class VOC12Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))
        # image = Image.open(img_name)

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label

def _transform_resize():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_shape=[224, 224],
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.resize_shape = resize_shape
        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.normalize = _transform_resize()
        self.scale = 1
        self.patch_size = 16

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        print('image', image.size)
        image = cv2.resize(image, (224, 224))
        
        img_box = None
        if self.aug:
            image = np.array(image)
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            '''
            if self.rescale_range:
                image = transforms.random_scaling(
                    image,
                    scale_range=self.rescale_range)
            
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            #image = self.color_jittor(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image,
                    crop_size=self.crop_size,
                    mean_rgb=[0,0,0],#[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
        '''
        
        image = transforms.normalize_img(image)
        # image = self.normalize(image)
        # image = image.numpy()
        
        # print('image', image.shape)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)
        
#         ori_height = image.size[1]
#         ori_width = image.size[0]
        
#         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
#         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
#         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
#         image = image.convert("RGB")

        original_image = image.copy()

        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, original_image, image, cls_label, img_box
        else:
            return img_name, original_image, image, cls_label


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.normalize = _transform_resize()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.scale = 1
        self.patch_size = 16
        

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            image = np.array(image)
            '''
            if self.resize_range: 
                image, label = transforms.random_resize(
                    image, label, size_range=self.resize_range)
            
            if self.rescale_range:
                image, label = transforms.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range)
            '''
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label, img_box = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    # mean_rgb=[123.675, 116.28, 103.53], 
                    ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        # image = self.normalize(image)
        # image = image.numpy()
        
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)
#         ori_height = image.size[1]
#         ori_width = image.size[0]
        
#         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
#         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
#         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
#         image = image.convert("RGB")

        image, label = self.__transforms(image=image, label=label)

        if self.stage=='test':
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label
