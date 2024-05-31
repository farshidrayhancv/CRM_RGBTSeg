import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

def get_image_set(images_modified_path, image_name):
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

    # Mean and standard deviation normalization of the thermal image
    thermal_mean = np.mean(thermal_image)
    thermal_std = np.std(thermal_image)
    thermal_image_mean_std_normalized = (thermal_image - thermal_mean) / thermal_std

    return rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized

def display_image_set(rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, modified_thermal_image):
    # Display the images
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

    ax[0].imshow(rgb_image)
    ax[0].set_title('RGB Image')
    ax[0].axis('off')

    ax[1].imshow(thermal_image, cmap='hot')
    ax[1].set_title('Thermal Image')
    ax[1].axis('off')

    ax[2].imshow(thermal_image_normalized, cmap='hot')
    ax[2].set_title('Thermal Image (Min-Max Normalized)')
    ax[2].axis('off')

    ax[3].imshow(thermal_image_mean_std_normalized, cmap='hot')
    ax[3].set_title('Thermal Image (Mean-Std Normalized)')
    ax[3].axis('off')

    ax[4].imshow(modified_thermal_image, cmap='hot')
    ax[4].set_title('Modified Thermal Image')
    ax[4].axis('off')

    plt.show()

def main(parent_folder, start_image_name):
    # Define paths
    images_path = os.path.join(parent_folder, 'images')
    images_modified_path = os.path.join(parent_folder, 'images_modified')
    

    # Construct the image file name
    image_name = f"{start_image_name}.png"

    # Get the image set
    rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized = get_image_set(images_path, image_name)
    if rgb_image is None:
        return

    # Load the modified thermal image
    modified_thermal_image_name = f"{start_image_name}.png"
    rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized = get_image_set(images_modified_path, image_name)
    
    # Display the image set
    display_image_set(rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, modified_thermal_image)

    # Return the images for further use
    return rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, modified_thermal_image

# Example usage
parent_folder = '/home/farshid/ComputerVisionDev/CRM_RGBTSeg/datasets/MFdataset'
start_image_name = '00001D'  # Replace with the desired starting image name
rgb_image, thermal_image, thermal_image_normalized, thermal_image_mean_std_normalized, modified_thermal_image = main(parent_folder, start_image_name)

# Now rgb_image, thermal_image, thermal_image_normalized, and thermal_image_mean_std_normalized are available for further use

