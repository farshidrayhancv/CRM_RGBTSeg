#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:09:08 2024

Author: Farshid
"""

import os
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt

def get_image_set(images_modified_path, image_name, t_values):
    image_path = os.path.join(images_modified_path, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found in {images_modified_path}.")
        return None, None, None, None

    # Load the image
    image = imageio.imread(image_path)

    # Separate the RGB and thermal channels
    rgb_image = image[:, :, :3]
    thermal_image = image[:, :, 3]

    # Min-max normalization of the thermal image
    thermal_image_normalized = (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image))

    # Mean and STD normalization of the thermal image
    mean_thermal = np.mean(thermal_image)
    std_thermal = np.std(thermal_image)
    thermal_image_mean_std_normalized = (thermal_image - mean_thermal) / std_thermal

    # Load modified images for each t value
    thermal_images_modified = []
    for t in t_values:
        modified_image_path = image_path.replace("images", f"images_modified_t{t}")
        if not os.path.exists(modified_image_path):
            print(f"Modified image {image_name} not found for t={t}.")
            return None, None, None, None

        modified_image = imageio.imread(modified_image_path)
        thermal_images_modified.append(modified_image[:, :, 3])

    return rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, thermal_images_modified

def display_image_set(image_sets, t_values):
    # Display the images stacked vertically
    fig, axs = plt.subplots(len(image_sets), 4 + len(t_values), figsize=(20, 15))

    # titles = ['RGB Image', 'Thermal Image', 'Thermal Image Min Max Normalised', 'Thermal Image Mean STD Normalised'] + [f'Thermal Image Modified t={t}' for t in t_values]

    # for j, title in enumerate(titles):
    #     axs[0, j].set_title(title)
    
    for i, (rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, thermal_images_modified) in enumerate(image_sets):
        axs[i, 0].imshow(rgb_image)
        axs[i, 0].axis('off')
        axs[i, 0].set_aspect('equal')

        axs[i, 1].imshow(thermal_image, cmap='hot')
        axs[i, 1].axis('off')
        axs[i, 1].set_aspect('equal')

        axs[i, 2].imshow(thermal_image_normalized, cmap='hot')
        axs[i, 2].axis('off')
        axs[i, 2].set_aspect('equal')
        
        axs[i, 3].imshow(thermal_image_mean_std_normalized, cmap='hot')
        axs[i, 3].axis('off')
        axs[i, 3].set_aspect('equal')
        
        for j, thermal_image_modified in enumerate(thermal_images_modified):
            axs[i, 4 + j].imshow(thermal_image_modified, cmap='hot')
            axs[i, 4 + j].axis('off')
            axs[i, 4 + j].set_aspect('equal')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# Function to read image names from train.txt
def get_image_names_from_file(file_path):
    with open(file_path, 'r') as file:
        image_names = [line.strip() + '.png' for line in file]
    return image_names

# Example usage
parent_folder = '/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/MFdataset'
train_file_path = os.path.join(parent_folder, 'train.txt')

# Read image names from train.txt
image_names = get_image_names_from_file(train_file_path)

# Randomly select five images
# random.seed()
selected_images = random.sample(image_names, 5)

# Define paths
images_modified_path = os.path.join(parent_folder, 'images')  # this is the unmodified image

# Define t values
t_values = [0.5, 1, 10, 5]

# Get the image sets for the selected images
image_sets = []
for image_name in selected_images:
    image_set = get_image_set(images_modified_path, image_name, t_values)
    if image_set:
        image_sets.append(image_set)

# Display the image sets
if image_sets:
    display_image_set(image_sets, t_values)
