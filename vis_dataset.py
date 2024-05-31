import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks

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

# Define dataset and parameters
dataset = "PSTdataset"  # Change this to "uclphydataset" or "MFdataset" if needed
subsample_factor = 5  # Adjust this factor as needed
image_id = 12  # For display purposes
hist_bin_number = 50  # Number of bins for finding peak

# Folder paths
image_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/thermal/"
output_folder = f"/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/{dataset}/thermal_modified/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of image file names
image_files = os.listdir(image_folder)
total_images = len(image_files)
image_files.sort()

# Get image dimensions
try:
    image_temp = np.load(os.path.join(image_folder, image_files[0]))  # Load the first image
    height, width = image_temp.shape
except:
    image_temp = Image.open(os.path.join(image_folder, image_files[0]))
    width, height = image_temp.size

# Initialize array to store all images
all_images_array = np.zeros((total_images, height, width), dtype=np.uint8)

# Load all images into the array
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    image_data = np.load(image_path).reshape((height, width)) if isinstance(image_temp, np.ndarray) else np.array(Image.open(image_path))
    all_images_array[i] = image_data

# Flatten the array for histogram calculation
all_images_flattened = all_images_array.flatten()

# Calculate histogram
hist, bins = np.histogram(all_images_flattened, bins=hist_bin_number)

# Find peak frequency temperature
peaks, _ = find_peaks(hist, height=0)
peak_temp_index = peaks[np.argmax(hist[peaks])]
peak_temp = (bins[peak_temp_index] + bins[peak_temp_index + 1]) / 2

# Remove temperatures close to the peak frequency temperature
tolerance = 10  # Define tolerance range
modified_images = [remove_peak_frequency(image, peak_temp, tolerance) for image in all_images_array]

# Scale the modified images
scaled_images = [scale_image(image) for image in modified_images]
scaled_images = np.array(scaled_images)

# Flatten the array for histogram calculation of scaled images
scaled_images_flattened = scaled_images.flatten()

# Calculate histogram for scaled images
hist_scaled, bins_scaled = np.histogram(scaled_images_flattened, bins=hist_bin_number)

# Calculate mean and std before and after scaling
mean_original = np.array([np.mean(image) for image in all_images_array])
std_original = np.array([np.std(image) for image in all_images_array])
mean_scaled = np.array([np.mean(image) for image in scaled_images])
std_scaled = np.array([np.std(image) for image in scaled_images])

# Subsample the data for plotting
rng_subsampled = np.arange(0, len(mean_original), subsample_factor)
mean_values_subsampled_original = mean_original[rng_subsampled]
std_values_subsampled_original = std_original[rng_subsampled]
mean_values_subsampled_scaled = mean_scaled[rng_subsampled]
std_values_subsampled_scaled = std_scaled[rng_subsampled]

# Plot mean values and std deviation for original images
plt.figure(figsize=(10, 6))
plt.plot(rng_subsampled, mean_values_subsampled_original, color='green', label='Mean')
plt.errorbar(rng_subsampled, mean_values_subsampled_original, std_values_subsampled_original, fmt='none', ecolor='red', elinewidth=1, label='Std Dev')
plt.title('Pixel/Temperature Value Distribution Mean and Std Original')
plt.xlabel('Image Index')
plt.ylabel('Average Pixel/Temperature Value')
plt.legend()
plt.tight_layout()
plt.show()

# Plot mean values and std deviation for scaled images
plt.figure(figsize=(10, 6))
plt.plot(rng_subsampled, mean_values_subsampled_scaled, color='green', label='Mean')
plt.errorbar(rng_subsampled, mean_values_subsampled_scaled, std_values_subsampled_scaled, fmt='none', ecolor='red', elinewidth=1, label='Std Dev')
plt.title('Pixel/Temperature Value Distribution Mean and Std Scaled')
plt.xlabel('Image Index')
plt.ylabel('Average Pixel/Temperature Value')
plt.legend()
plt.tight_layout()
plt.show()

# Plot original histogram
plt.figure(figsize=(10, 6))
plt.hist(all_images_flattened, bins=hist_bin_number, color='blue', alpha=0.7, label='Original Pixel/Temperature Values')
plt.plot(bins[peaks], hist[peaks], "o", color='orange', label='Original Peaks')
plt.title('Original Pixel/Temperature Value Distribution')
plt.xlabel('Pixel/Temperature Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Plot modified histogram
plt.figure(figsize=(10, 6))
plt.hist(scaled_images_flattened, bins=hist_bin_number, color='green', alpha=0.7, label='Modified Pixel/Temperature Values')
plt.plot(bins_scaled[peaks], hist_scaled[peaks], "o", color='orange', label='Scaled Peaks')
plt.title('Pixel/Temperature Value Distribution Modified')
plt.xlabel('Pixel/Temperature Value')
plt.ylabel('Frequency')
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
