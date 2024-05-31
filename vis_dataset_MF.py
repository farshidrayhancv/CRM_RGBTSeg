#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 03:52:30 2024

@author: farshid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks


def scale_image(image, peak_value):
    # Scale image so that the highest peak becomes 255 and lowest becomes 0
    ep =  0.00001
    max_value = np.max(image)
    min_value = np.min(image)
    scaled_image = ((image - min_value) / (max_value - min_value + ep)) * peak_value
    return scaled_image.astype(np.uint8)

if __name__ == "__main__":
    
    
    dataset = "uclphydataset"  # Change this to "PSTdataset" if needed
    dataset = "PSTdataset"  # Change this to "PSTdataset" if needed
    dataset = "MFdataset"  # Change this to "PSTdataset" if needed
    
    subsample_factor = 4096  # Adjust this factor as needed
    image_id = 12 # For display purpose
    hist_bin_number = 50 # Number of bins for finding peak
    
    image_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/separated_images/"
    output_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/thermal_modified/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Get list of image file names
    image_files = os.listdir(image_folder)
    total_images = len(image_files)
    # Sort image files for consistency
    
    image_files =  [item for item in image_files if "_th" not in item]
    image_files.sort()
    


    # Get image dimensions
    try:
        image_temp = np.load(os.path.join(image_folder, image_files[0]))  # Load the first image
        height, width = image_temp.shape
    except:
        image_temp = Image.open(os.path.join(image_folder, image_files[0].replace("_rgb", "_th")))
        height, width = image_temp.size

    # Initialize array to store all images
    all_images_array = np.zeros((total_images, width, height), dtype=np.uint8)

    # Load all images into the array
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file).replace("_rgb", "_th")
        image_data = np.load(image_path).reshape((width, height)) if isinstance(image_temp, np.ndarray) else np.array(Image.open(image_path))
        all_images_array[i] = image_data

    # Flatten the array for mean and std calculations
    all_images_array_flattened = all_images_array.reshape((total_images, -1))

    # Calculate mean and std across all images
    mean_values = np.mean(all_images_array_flattened, axis=0)
    std_values = np.std(all_images_array_flattened, axis=0)

    # Subsample the data for plotting
    rng_subsampled = np.arange(0, len(mean_values), subsample_factor)
    mean_values_subsampled = mean_values[rng_subsampled]
    std_values_subsampled = std_values[rng_subsampled]

    # Plot mean values in green
    plt.figure(figsize=(10, 6))
    plt.plot(rng_subsampled, mean_values_subsampled, color='green', label='Mean')
    plt.errorbar(rng_subsampled, mean_values_subsampled, std_values_subsampled, fmt='none', ecolor='red', elinewidth=1, label='Std Dev')
    plt.title('Pixel/Temperature Value Distribution Mean and Std Original')
    plt.xlabel('Pixel/Temperature Position')
    plt.ylabel('Average Pixel/Temperature Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot histogram to visualize distribution
    plt.figure(figsize=(10, 6))
    hist, bins, _ = plt.hist(all_images_array_flattened.flatten(), bins=50, color='blue', alpha=0.7, label='Pixel/Temperature Values')
    peaks, _ = find_peaks(hist, height=0)  # Identify peaks
    plt.plot(bins[peaks], hist[peaks], "o", color='orange', label='Scaled Peaks')
    
    plt.title('Pixel/Temperature Value Distribution Modified')
    plt.xlabel('Pixel/Temperature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    




    
     # Scale all images in the array based on the highest peak value
    peak_value = np.max(all_images_array)
    scaled_images = [scale_image(image, peak_value) for image in all_images_array]
    scaled_images = np.array(scaled_images)
    
    
    # Plot histogram to visualize distribution
    plt.figure(figsize=(10, 6))
    hist_scaled, bins_scaled, _ = plt.hist(scaled_images.flatten(), bins=hist_bin_number, color='blue', alpha=0.7, label='Pixel/Temperature Values')
    peaks_scaled, _ = find_peaks(hist_scaled, height=0)  # Identify peaks

    # Flatten the array for mean and std calculations
    scaled_images_flattened = scaled_images.reshape((total_images, -1))

    # Calculate mean and std across all images
    mean_values_scaled = np.mean(scaled_images_flattened, axis=0)
    std_values_scaled = np.std(scaled_images_flattened, axis=0)

    # Subsample the data for plotting
    rng_subsampled_scaled = np.arange(0, len(mean_values_scaled), subsample_factor)
    mean_values_subsampled_scaled = mean_values_scaled[rng_subsampled_scaled]
    std_values_subsampled_scaled = std_values_scaled[rng_subsampled_scaled]
    
    
    
    # Highlight peaks on histogram
    plt.plot(bins_scaled[peaks_scaled], hist_scaled[peaks_scaled], "o", color='orange', label='Scaled Peaks')
    plt.title('Pixel/Temperature Value Distribution Modified')
    plt.xlabel('Pixel/Temperature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    # Plot mean values in green
    plt.figure(figsize=(10, 6))
    plt.plot(rng_subsampled_scaled, mean_values_subsampled, color='green', label='Mean')
    plt.errorbar(rng_subsampled_scaled, mean_values_subsampled_scaled, std_values_subsampled_scaled, fmt='none', ecolor='red', elinewidth=1, label='Std Dev')
    plt.title('Pixel/Temperature Value Distribution Mean & Std Modifeid')
    plt.xlabel('Pixel/Temperature Position')
    plt.ylabel('Average Pixel/Temperature Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # Display original image and the first modified image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(all_images_array[image_id], cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(scaled_images[image_id], cmap='gray')
    plt.title('Modified Image')


    plt.tight_layout()
    plt.show()
    
    # # Load all images into the array
    for image_file, scaled_image in zip(image_files, scaled_images):


        # Save the image in the original format
        output_path = os.path.join(output_folder, image_file)
        np.save(output_path, scaled_image) if isinstance(image_temp, np.ndarray) else image_temp.save(output_path)
    
    
    
