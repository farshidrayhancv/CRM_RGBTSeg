#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:51:11 2024

@author: farshid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from scipy.fft import fft2, ifft2
import imageio

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

def scale_image(image):
    """
    Scale image so that the highest value becomes 255 and lowest becomes 0.
    
    Args:
    - image (np.ndarray): The input image.
    
    Returns:
    - np.ndarray: The scaled image.
    """
    ep = 1e-5
    max_value = np.max(image)
    min_value = np.min(image)
    scaled_image = ((image - min_value) / (max_value - min_value + ep)) * 255
    return scaled_image.astype(np.uint8)

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
    
    # Manipulate frequency components (here we simply zero out high frequencies for demonstration)
    rows, cols = freq_image.shape
    crow, ccol = rows // 2 , cols // 2
    freq_image[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    # Transform back to spatial domain
    norm_image = ifft2(freq_image).real
    
    # Scale the image to the range [0, 255]
    return scale_image(norm_image)

# Define dataset and image folder
dataset = "MFdataset"  # Change this to "PSTdataset" if needed

image_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/separated_images/"
output_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/images_modified_fourier_t/"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of image file names, excluding ones with "_th"
image_files = [f for f in os.listdir(image_folder) if "_th" not in f]
image_files.sort()

# Get image dimensions
try:
    image_temp = np.load(os.path.join(image_folder, image_files[0]))  # Load the first image
    height, width = image_temp.shape
except:
    image_temp = Image.open(os.path.join(image_folder, image_files[0].replace("_rgb", "_th")))
    width, height = image_temp.size

# Load all images into the array
for idx, image_file in enumerate(image_files):
    # Load RGB image
    rgb_image_path = os.path.join(image_folder, image_file)
    rgb_image = np.array(Image.open(rgb_image_path))
    
    # Load thermal image
    th_image_path = os.path.join(image_folder, image_file.replace("_rgb", "_th"))
    th_image = np.load(th_image_path).reshape((height, width)) if isinstance(image_temp, np.ndarray) else np.array(Image.open(th_image_path))
    
    # Remove temperatures close to the peak frequency temperature
    hist, bins = np.histogram(th_image.flatten(), bins=50)
    peaks, _ = find_peaks(hist, height=0)
    peak_temp_index = peaks[np.argmax(hist[peaks])]
    peak_temp = (bins[peak_temp_index] + bins[peak_temp_index + 1]) / 2
    tolerance = 0.5  # Define tolerance range
    modified_th_image = remove_peak_frequency(th_image, peak_temp, tolerance)
    
    # Apply frequency-based normalization
    norm_th_image = frequency_based_normalization(modified_th_image)
    
    # Create 4-channel image
    four_channel_image = np.zeros((height, width, 4), dtype=np.uint8)
    four_channel_image[:, :, :3] = rgb_image
    four_channel_image[:, :, 3] = norm_th_image
    
    # Save the 4-channel image as PNG
    output_filename = f"{image_file.replace('_rgb', '')}"
    output_path = os.path.join(output_folder, output_filename)
    imageio.imwrite(output_path, four_channel_image)
    
    # Flip the images horizontally
    flipped_rgb_image = np.fliplr(rgb_image)
    flipped_th_image = np.fliplr(th_image)
    
    # Create 4-channel flipped image
    four_channel_flipped_image = np.zeros((height, width, 4), dtype=np.uint8)
    four_channel_flipped_image[:, :, :3] = flipped_rgb_image
    four_channel_flipped_image[:, :, 3] = frequency_based_normalization(remove_peak_frequency(flipped_th_image, peak_temp, tolerance))
    
    # Save the flipped 4-channel image as PNG
    output_filename_flip = f"{image_file.replace('_rgb', '_flip')}"
    output_path_flip = os.path.join(output_folder, output_filename_flip)
    imageio.imwrite(output_path_flip, four_channel_flipped_image)

print("Images modified and saved successfully.")
