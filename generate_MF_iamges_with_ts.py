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
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft2, ifft2

def remove_peak_frequency(image, peak_temp, tolerance=10):
    """
    Remove temperatures close to the peak frequency temperature from the image.
    
    Args:
    - image (np.ndarray): The input image.
    - peak_temp (float): The peak frequency temperature.
    - tolerance (float): The tolerance range to remove around the peak frequency temperature.
    
    Returns:
    - np.ndarray: The modified image.
    """
    mask = (image < peak_temp - tolerance) | (image > peak_temp + tolerance)
    return image * mask

def frequency_based_normalization(image):
    """
    Perform frequency-based normalization on the image.
    
    Args:
    - image (np.ndarray): The input image.
    
    Returns:
    - np.ndarray: The normalized image.
    """
    # Transform image to frequency domain
    freq_image = fft2(image)
    
    # Manipulate frequency components (here we zero out high frequencies for demonstration)
    rows, cols = freq_image.shape
    crow, ccol = rows // 2 , cols // 2
    freq_image[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    # Transform back to spatial domain
    norm_image = ifft2(freq_image).real
    
    # Scale the image to the range [0, 255]
    norm_image = (norm_image - norm_image.min()) / (norm_image.max() - norm_image.min()) * 255
    return norm_image.astype(np.uint8)

def get_image_set(images_path, image_name):
    image_path = os.path.join(images_path, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found in {images_path}.")
        return None, None

    # Load the image
    image = imageio.imread(image_path)

    # Separate the RGB and thermal channels
    rgb_image = image[:, :, :3]
    thermal_image = image[:, :, 3]

    return rgb_image, thermal_image

def calculate_peak_frequency(thermal_images):
    # Flatten all thermal images and concatenate them
    all_temperatures = np.concatenate([img.flatten() for img in thermal_images])
    
    # Calculate the histogram of the temperature values
    histogram, bin_edges = np.histogram(all_temperatures, bins=256)
    
    # Find the bin with the maximum frequency
    peak_bin_index = np.argmax(histogram)
    
    # Calculate the peak temperature
    peak_temp = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
    return peak_temp

def display_image_set(image_sets, peak_temp, tolerances):
    # Display the images stacked vertically
    fig, axs = plt.subplots(len(image_sets), len(tolerances) + 1, figsize=(20, 15))

    for i, (rgb_image, thermal_image) in enumerate(image_sets):
        # Original thermal image plot
        axs[i, 0].imshow(thermal_image, cmap='hot')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Original Thermal Image')

        for j, tolerance in enumerate(tolerances):
            modified_image = remove_peak_frequency(thermal_image, peak_temp, tolerance)
            norm_image = frequency_based_normalization(modified_image)
            axs[i, j + 1].imshow(norm_image, cmap='hot')
            axs[i, j + 1].axis('off')
            axs[i, j + 1].set_title(f'Modified t={tolerance}')

    plt.tight_layout()
    plt.show()

    # Display the 3D plots for the original and modified thermal images
    for rgb_image, thermal_image in image_sets:
        fig = plt.figure(figsize=(20, 6))

        X, Y = np.meshgrid(range(thermal_image.shape[1]), range(thermal_image.shape[0]))

        # Original thermal image plot
        ax1 = fig.add_subplot(151, projection='3d')
        ax1.plot_surface(X, Y, thermal_image, cmap='viridis')
        ax1.set_title('Original Thermal Image')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Temperature')

        for j, tolerance in enumerate(tolerances):
            modified_image = remove_peak_frequency(thermal_image, peak_temp, tolerance)
            norm_image = frequency_based_normalization(modified_image)
            ax = fig.add_subplot(152 + j, projection='3d')
            ax.plot_surface(X, Y, norm_image, cmap='viridis')
            ax.set_title(f'Modified t={tolerance}')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Temperature')

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

# Define paths
images_path = os.path.join(parent_folder, 'images')

# Load all thermal images to calculate peak frequency
thermal_images = []
for image_name in image_names:
    _, thermal_image = get_image_set(images_path, image_name)
    if thermal_image is not None:
        thermal_images.append(thermal_image)

# Calculate peak frequency temperature from all images
peak_temp = calculate_peak_frequency(thermal_images)

# Randomly select five images
random.seed() #42
selected_images = random.sample(image_names, 5)

# Get the image sets for the selected images
image_sets = []
for image_name in selected_images:
    image_set = get_image_set(images_path, image_name)
    if image_set:
        image_sets.append(image_set)

# Define tolerances
tolerances = [10, 5, 1, 0.5]

# Display the image sets and 3D plots
if image_sets:
    display_image_set(image_sets, peak_temp, tolerances)
