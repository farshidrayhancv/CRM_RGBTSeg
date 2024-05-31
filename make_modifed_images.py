#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 02:30:54 2024

@author: farshid
"""

import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Example usage
parent_folder = '/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/MFdataset'

# Define paths
separated_images_path = os.path.join(parent_folder, 'separated_images')
images_modified_path = os.path.join(parent_folder, 'images_modified')

# Create the output directory if it doesn't exist
os.makedirs(images_modified_path, exist_ok=True)

# Process each image in the separated_images folder
for image_name in os.listdir(separated_images_path):
    if not image_name.endswith('_rgb.png'):
        continue

    # Remove '_rgb' from the image name to get the base name
    base_name = image_name.replace('_rgb.png', '')

    # Build full paths to the current separated image and the corresponding thermal image
    separated_image_path = os.path.join(separated_images_path, image_name)
    thermal_image_path = os.path.join(separated_images_path, base_name + '_th.png')
    
    # Check if the corresponding thermal image exists
    if os.path.exists(thermal_image_path):
        # Read the separated image and thermal image
        separated_image = imageio.imread(separated_image_path)
        thermal_image = imageio.imread(thermal_image_path)
        
        # Ensure the thermal image is single channel (grayscale)
        if len(thermal_image.shape) == 3 and thermal_image.shape[2] == 3:
            thermal_image = np.mean(thermal_image, axis=2).astype(separated_image.dtype)

        # Expand thermal image dimensions to match the separated image
        thermal_image_expanded = np.expand_dims(thermal_image, axis=2)
        
        # Combine separated image and thermal image as an additional channel
        modified_image = np.concatenate((separated_image, thermal_image_expanded), axis=2)
        
        # Save the modified image
        modified_image_path = os.path.join(images_modified_path, base_name + '.png')
        # imageio.imwrite(modified_image_path, modified_image)
        print(f'Saved modified image: {modified_image_path}')
        
        # Create the flipped version of the separated image and thermal image
        separated_image_flipped = np.fliplr(separated_image)
        thermal_image_flipped = np.fliplr(thermal_image)
        thermal_image_flipped_expanded = np.expand_dims(thermal_image_flipped, axis=2)
        modified_image_flipped = np.concatenate((separated_image_flipped, thermal_image_flipped_expanded), axis=2)
        
        # Save the flipped modified image
        modified_image_flipped_path = os.path.join(images_modified_path, base_name + '_flip.png')
        # imageio.imwrite(modified_image_flipped_path, modified_image_flipped)
        print(f'Saved modified flipped image: {modified_image_flipped_path}')
        
        # Display the images side by side using matplotlib
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Min-max normalization
        thermal_image_normalized = (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image))
        thermal_image_flipped_normalized = (thermal_image_flipped - np.min(thermal_image_flipped)) / (np.max(thermal_image_flipped) - np.min(thermal_image_flipped))
        
        # Plot the images
        ax[0].imshow(separated_image)
        ax[0].set_title('RGB Image')
        ax[0].axis('off')
        
        ax[1].imshow(thermal_image, cmap='hot')
        ax[1].set_title('Thermal Image')
        ax[1].axis('off')
        
        ax[2].imshow(thermal_image_normalized, cmap='hot')
        ax[2].set_title('Normalised Thermal Image')
        ax[2].axis('off')
        
        plt.show()
        
        break
        
    else:
        print(f'Thermal image for {base_name} not found, skipping.')
        
        
        
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Parent folder
parent_folder = '/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/MFdataset'

# Paths to thermal and separated images folders
thermal_folder = os.path.join(parent_folder, 'thermal')
separated_images_folder = os.path.join(parent_folder, 'separated_images')

# List of image files
thermal_images = os.listdir(thermal_folder)
separated_images = os.listdir(separated_images_folder)

# Iterate through the images and compare
for image_name in thermal_images:
    # Check if corresponding separated image exists
    separated_image_name = image_name.replace('.png', '_th.png')
    if separated_image_name in separated_images:
        # Load images
        thermal_image_path = os.path.join(thermal_folder, image_name)
        separated_image_path = os.path.join(separated_images_folder, separated_image_name)
        thermal_image = imageio.imread(thermal_image_path)
        separated_image = imageio.imread(separated_image_path)
        
        # Min-max normalization on thermal image
        thermal_image_normalized = (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image))
        
        # Calculate absolute difference
        difference = abs(thermal_image - separated_image)
        
        # Plot original thermal, separated, and normalized thermal images along with their difference
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(thermal_image, cmap='gray')
        axes[0].set_title('Thermal Image')
        axes[0].axis('off')
        axes[1].imshow(separated_image, cmap='gray')
        axes[1].set_title('Separated Image')
        axes[1].axis('off')
        axes[2].imshow(difference, cmap='gray')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        axes[3].imshow(thermal_image_normalized, cmap='gray')
        axes[3].set_title('Normalized Thermal Image')
        axes[3].axis('off')
        plt.show()

        
        
