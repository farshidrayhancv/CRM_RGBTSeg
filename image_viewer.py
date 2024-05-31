#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:49:13 2024

@author: farshid
"""



import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# def load_random_images(root_folder, num_images=3):
#     rgb_folder = os.path.join(root_folder, 'rgb')
#     thermal_folder = os.path.join(root_folder, 'thermal')
#     label_folder = os.path.join(root_folder, 'labels')

#     # Get list of images in each folder
#     rgb_images = os.listdir(rgb_folder)
#     thermal_images = [img.replace("jpg", "npy") for img in rgb_images]
#     label_images = [img.replace("jpg", "png") for img in rgb_images]

#     if not rgb_images:
#         print("No RGB images found.")
#         return

#     # Choose random images
#     rgb_images_sample = random.sample(rgb_images, num_images)
#     thermal_images_sample = [img.replace("jpg", "npy") for img in rgb_images_sample]
#     label_images_sample = [img.replace("jpg", "png") for img in rgb_images_sample]

#     # Load random images
#     rgb_image_samples = [Image.open(os.path.join(rgb_folder, img)) for img in rgb_images_sample]
#     thermal_image_samples = [np.load(os.path.join(thermal_folder, img)) for img in thermal_images_sample]
#     label_image_samples = [Image.open(os.path.join(label_folder, img)) for img in label_images_sample]

#     return rgb_image_samples, thermal_image_samples, label_image_samples

# def display_images(rgb_images, thermal_images, label_images):
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))

#     for i in range(3):
#         # Display RGB image
#         axs[i, 0].imshow(rgb_images[i])
#         axs[i, 0].set_title('RGB Image')
#         axs[i, 0].axis('off')

#         # Display Thermal image
#         axs[i, 1].imshow(thermal_images[i])
#         axs[i, 1].set_title('Thermal Image')
#         axs[i, 1].axis('off')

#         # Display Label image
#         axs[i, 2].imshow(label_images[i])
#         axs[i, 2].set_title('Label Image')
#         axs[i, 2].axis('off')

#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    
    # Choose random images
    # random.seed(23)    
    dataset = "PST" # PST or UCL
    number_of_images = 5
    
    if dataset == "PST":
        root_folder = "/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/PSTdataset/train"
    else:
        root_folder = "/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/Segmentation2"
    

    
    rgb_folder = os.path.join(root_folder, 'rgb')
    thermal_folder = os.path.join(root_folder, 'thermal')
    label_folder = os.path.join(root_folder, 'labels')

    # Get list of images in each folder
    rgb_images = os.listdir(rgb_folder)
    thermal_images = [img.replace("jpg", "npy").replace("bmp", "npy").replace("png", "npy") for img in rgb_images]
    label_images = [img.replace("jpg", "png").replace("bmp", "png") for img in rgb_images]

    if not rgb_images:
        print("No RGB images found.")
        # return


    rgb_images_sample = random.sample(rgb_images, number_of_images)
    if dataset == "PST":
        thermal_images_sample = [img for img in rgb_images_sample]
    else:
        thermal_images_sample = [img.replace("jpg", "npy").replace("bmp", "npy").replace("png", "npy") for img in rgb_images_sample]
        
    label_images_sample = [img.replace("jpg", "png").replace("bmp", "png") for img in rgb_images_sample]

    # Load random images
    rgb_image_samples = [Image.open(os.path.join(rgb_folder, img)) for img in rgb_images_sample]
    if dataset == "PST":
        thermal_image_samples = [Image.open(os.path.join(thermal_folder, img)) for img in thermal_images_sample]
    else:
        thermal_image_samples = [np.load(os.path.join(thermal_folder, img)) for img in thermal_images_sample]
    
    label_image_samples = [Image.open(os.path.join(label_folder, img)) for img in label_images_sample]

    if rgb_images:
        fig, axs = plt.subplots(3, number_of_images, figsize=(15, 15))
        # axs[0, 0].set_title('RGB Image')
        # axs[0, 1].set_title('Thermal Image')
        # axs[0, 2].set_title('Label')
        for i in range(number_of_images):
            # Display RGB image
            axs[ 0, i].imshow(rgb_image_samples[i])
            # axs[i, 0].set_title('RGB Image')
            axs[0, i].axis('off')

            # Display Thermal image
            axs[1, i].imshow(thermal_image_samples[i])
            # axs[i, 1].set_title('Thermal Image')
            axs[ 1, i].axis('off')

            # Display Label image
            axs[2, i].imshow(label_image_samples[i])
            # axs[i, 2].set_title('Label Image')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.show()
