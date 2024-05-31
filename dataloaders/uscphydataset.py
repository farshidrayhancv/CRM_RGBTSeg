# Written by Ukcheol Shin, Jan. 24, 2023 using the following two repositories.
# PST900: https://github.com/ShreyasSkandanS/pst900_thermal_rgb
# Mask2Former: https://github.com/facebookresearch/Mask2Former

import cv2
import numpy as np
import os, torch
from skimage.transform import resize
from imageio import imread
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from detectron2.structures import BitMasks, Instances
from detectron2.data import transforms as T
from .augmentation import ColorAugSSDTransform, MaskGenerator

class Uclphy_dataset(Dataset):

    def __init__(self, data_dir, cfg, split):
        super(Uclphy_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], \
            'split must be "train"|"val"|"test"' 

        # read dataset list, all files have the same name across 'rgb', 'label', 'thermal', 'depth' folders
        self.data_list = os.listdir(os.path.join(data_dir, 'rgb')) 
        self.data_list.sort()

        self.data_dir  = os.path.join(data_dir, split)
        self.split     = split
        self.n_data    = len(self.data_list)
        self.size_divisibility = -1
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        self.augmentations = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            self.augmentations.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            self.augmentations.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        self.augmentations.append(T.RandomFlip())

        if cfg.INPUT.MASK.ENABLED:
            self.mask_generator = MaskGenerator(input_size=cfg.INPUT.MASK.SIZE, \
                                                mask_patch_size=cfg.INPUT.MASK.PATCH_SIZE, \
                                                model_patch_size=cfg.MODEL.SWIN.PATCH_SIZE, \
                                                mask_ratio=cfg.INPUT.MASK.RATIO,
                                                mask_type=cfg.INPUT.MASK.TYPE,
                                                strategy=cfg.INPUT.MASK.STRATEGY
                                                )
        else:
            self.mask_generator = None 
            
    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image     = imread(file_path).astype('float32')
        return image

    def __getitem__(self, index):
        name  = self.data_list[index]
        image_rgb = self.read_image(name, 'rgb')
        name = name.replace('jpg', 'npy')
        name = name.replace('png', 'npy')
        name = name.replace('bmp', 'npy')
        
        
        # print(name)
        thermal = np.load(self.data_dir + '/thermal_modified/' + name )
        # thermal = (thermal - thermal.min()) / (thermal.max() -thermal.min())
        

        # name = name.replace('npy', 'jpg')
        name = name.replace('npy', 'png')

        image_rgb = resize(image_rgb, (512, 640, 3)).astype("float32")

        # print( name, image_rgb.shape, type(image_rgb), thermal.shape, type(thermal))

        image_thr = np.expand_dims(thermal, axis=2)
        image = np.concatenate((image_rgb,image_thr),axis=2).astype("float32")
        # depth = self.read_image(name, 'depth')

        sem_seg_gt = self.read_image(name, 'labels').astype("float32")
        sem_seg_gt = resize(sem_seg_gt, (512, 640)).astype("float32")

        color_to_integer = {(0, 0, 0): 0, (0, 0, 255): 1, (255, 0, 0): 2}
    
        # Replace RGB values with corresponding unique integers
        sem_seg_gt = np.array([color_to_integer.get(tuple(color), 0) for color in sem_seg_gt.reshape(-1, 3)]).reshape(sem_seg_gt.shape[:2])
    

        # unique_colors, color_indices = np.unique(sem_seg_gt.reshape(-1, 3), axis=0, return_inverse=True)
    
        # # Create a mapping dictionary from unique colors to unique integers
        # color_to_integer = {tuple(color): index for index, color in enumerate(unique_colors)}
    
        # # Replace RGB values with corresponding unique integers
        # sem_seg_gt = np.array([color_to_integer[tuple(color)] for color in sem_seg_gt.reshape(-1, 3)]).reshape(sem_seg_gt.shape[:2]).astype("float32")
    


        # print(image_rgb.shape, image_thr.shape, image.shape)

        # Data Augmentation
        if self.split == 'train':
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.augmentations, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image      = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("float32"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image      = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Packing data
        result = {}
        result["name"]  = name
        result["image"] = image
        result["sem_seg_gt"] = sem_seg_gt.long()

        # # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)
            
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:

                # for x in masks:
                #     x = x.copy()
                #     x = np.ascontiguousarray(x)
                #     x = torch.from_numpy(x)
                #     print(x.shape)
                
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                    
                # print(masks.tensor.type)
                instances.gt_masks = masks.tensor

            

            result["instances"] = instances

            

        # Prepare mask
        if (self.split == 'train') and (self.mask_generator is not None):
            mask1, mask2 = self.mask_generator()
            result["mask"] = torch.as_tensor(np.stack([mask1, mask2], axis=0))

        return result

    def __len__(self):
        return self.n_data
