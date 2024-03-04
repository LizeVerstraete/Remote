from __future__ import annotations

import csv
import cv2
import os

from sklearn.neighbors import KNeighborsClassifier

import openslide
from sklearn.cluster import KMeans
from PIL import Image
import math


# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import shutil
import warnings
from pathlib import Path

#import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage import color, exposure, measure, morphology

from tiatoolbox import logger
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.registration.wsi_registration import (
    AffineWSITransformer,
    DFBRegister,
    apply_bspline_transform,
    estimate_bspline_transform,
    match_histograms,
)
from tiatoolbox.wsicore.wsireader import WSIReader


def load_image(path, show_raw_image=False):
    """
    function to load the mrxs file
    :param path: where the .mrxs file is located
    :param show_raw_image: True if you want a plot of the original image (thumbnail of it)
    :return: an opened version of the slide which you can use for further computations
    """

    slide = openslide.open_slide(path)

    # plot the original image if requested
    if show_raw_image:
        # Get a thumbnail image (level 0) for display
        thumbnail = np.array(slide.get_thumbnail((1000, 1000)))
        # Display the image
        plt.figure()
        plt.imshow(thumbnail)
        plt.title('slide')
        plt.show()
    return slide


def crop_image(slide, level, show_cropping=False):
    """
    function that removes the white area around the biopsy leaving a rectangular image behind
    :param slide: the biopsy slide that needs to be cropped
    :param show_cropping: True if you want to visualize the biopsy after cropping
    :return: a cropped version of the image
    """
    # convert image to NumPy array
    # level 5 used since the whole image took too much memory to convert
    slide_np = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))

    # find where the image is located
    x, y = np.where((slide_np[:, :, 0] != 0) | (slide_np[:, :, 1] != 0) | (slide_np[:, :, 2] != 0))

    # find image boundaries
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    # crop the image within the boundaries
    # cropped_image = Image.fromarray(slide_np).crop((xmin, ymin, xmax, ymax))
    cropped_image = slide_np[xmin:xmax, ymin:ymax, :]

    location = [xmin, xmax, ymin, ymax]

    # plot the cropped image
    if show_cropping:
        plt.figure()
        plt.imshow(cropped_image)
        plt.title('cropped image')
        plt.show()

    # now crop the image at level 1 (assumed to be zoomlevel 20), but problem is takes too much memory, so not possible yet, TAKE THIS INTO ACCOUNT WHEN LOOKING AT XMIN ETC VALUES!!
    # we know that one pixel in level 5 contains 16 pixels in level 1
    # transformation of coordinates
    # xminHE *= 16
    # yminHE *= 16
    # xmaxHE *= 16
    # ymaxHE *= 16
    #
    # cropped_HE_level1 = slide_HE.read_region((xminHE,yminHE), 1, (xmaxHE-xminHE,ymaxHE-yminHE))
    #
    # cropped_HE_level1_np = np.array(cropped_HE_level1)
    #
    # plt.figure()
    # plt.imshow(cropped_HE_level1_np)
    # plt.title('cropped HE level 1')
    # plt.show()
    return cropped_image, location


def split_in_tiles(slide_to_split, original_slide, tile_size, level, location):
    """
    function that splits the given slide up in tiles of the requested size given in micrometers, the last tiles might be smaller due to non-divisibility
    :param slide_to_split: the slide that needs to be split up in tiles
    :param original_slide: the full slide before cropping
    :param tile_size: the size of the wanted tiles in micrometer
    :param level: the level from where the tiles are taken
    :param location: the position of the cropped image in the whole
    :return: the tiles of the wanted size where the last tiles might be smaller due to non-divisibility of the size of the image by the size of the tiles
    """

    #OTHER TILESPLITTING METHOD:
    # from openslide.deepzoom import DeepZoomGenerator
    #
    # # Generate object for tiles using the DeepZoomGenerator
    # tiles = DeepZoomGenerator(slide, tile_size=100, overlap=0, limit_bounds=False)

    # transform the tile_size to a number of pixels
    # we are still working in level 5 at the moment
    # first get the pixel size of level 0, the one at level 5 is 2^5 (=32) times as big
    pixel_size_level_0 = float(original_slide.properties.get('openslide.mpp-x'))
    pixel_size_level_n = pixel_size_level_0 * pow(2, level)
    nb_pixels_tile = math.ceil(tile_size / pixel_size_level_n)

    # split the slide in tiles
    pixels_x, pixels_y, _ = slide_to_split.shape
    nb_tiles_x = math.ceil(pixels_x / nb_pixels_tile)
    nb_tiles_y = math.ceil(pixels_y / nb_pixels_tile)

    # this works only if the cropped image is loaded in the same level previously as the one requested here

    tiles = []
    for i in range(nb_tiles_x):
        for j in range(nb_tiles_y):
            xmin = i * nb_pixels_tile
            xmax = min((i + 1) * nb_pixels_tile, pixels_x)
            ymin = j * nb_pixels_tile
            ymax = min((j + 1) * nb_pixels_tile, pixels_y)
            #tile = original_slide.read_region((location[0]+xmin,location[2]+ymin),level,((xmax-xmin),(ymax-ymin)))
            tile = slide_to_split[xmin:xmax, ymin:ymax, :]
            tiles.append(tile)

    # I think this might be the better way, but memory problems

    # for i in range(location[0],location[1],nb_pixels_tile):
    #     for j in range(location[2],location[3],nb_pixels_tile):
    #         tile = original_slide.read_region((i,j),level,(min(pixels_x,(location[1]-i)),min(pixels_y,(location[3]-j))))
    #         tiles.append(tile)
    return tiles

def safe_tiles(tiles, level):
    cols, rows = tiles.level_tiles[level]
    tile_dir = r'\esat\biomeddata\kkontras\r0786880\data\tiles'
    for row in range(rows):
        for col in range(cols):
            temp_tile = tiles.get_tile(level, (col, row))
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            if np.sum(np.any(temp_tile_np != np.array([255, 255, 255]), axis=-1)) > (
                    temp_tile_np.shape[0] * temp_tile_np.shape[
                1] / 2):  # only safe tile if more than half of it is not white
                tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
                print("Now saving tile with title: ", tile_name)
                plt.imsave(tile_name + ".png", temp_tile_np)


def label_tiles(tiles):
    """
    function that returns labels for the given tiles, for this function to return usefull information it should be fed
    with tiles from the stain image and not the H&E image
    :param tiles: the tiles for which labeling should be performed
    :return: the labels corresponding to each tile
    """
    labeled_tiles = []

    # assign labels to certain values inside the tiles NOTE: these values still need to be optimized!
    for i in range(len(tiles)):
        tile = tiles[i]
        nb_green = 0
        nb_purple = 0
        nb_none = 0
        pixels_x_tile, pixels_y_tile, _ = tile.shape
        for j in range(pixels_x_tile):
            for k in range(pixels_y_tile):
                if (tile[j][k][0] <= 50) and (150 <= tile[j][k][1] <= 255) and (tile[j][k][2] <= 50):
                    nb_green += 1
                elif (100 <= tile[j][k][0]) and (tile[j][k][1] <= 50) and (100 <= tile[j][k][2]):
                    nb_purple += 1
                else:
                    nb_none += 1
        # mean_value_tile = tiles[i].mean()
        # if mean_value_tile < 200:
        #     label = 'MUC5'
        # elif mean_value_tile < 100:
        #     label = 'CDX2'
        # else:
        #     label = 'mixed'
        if nb_green > 2*nb_purple:
            label = 'MUC'
        elif nb_purple > 2*nb_green:
            label = 'CDX2'
        else:
            label = 'mixed'
        labeled_tiles.append((tiles[i], label))
    return labeled_tiles

def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Pre-process image for registration using masks.

    This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.

    """
    image = color.rgb2gray(image)
    image = exposure.rescale_intensity(
        image,
        in_range=tuple(np.percentile(image, (0.5, 99.5))),
    )
    image = image * 255
    return image.astype(np.uint8)


def post_processing_mask(mask: np.ndarray) -> np.ndarray:
    """Post-process WSI masks."""
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((3, 3))).astype(int)

    # num of unique objects for segmentation is 2.
    num_unique_labels = 2
    # remove all the objects while keep the biggest object only
    label_img = measure.label(mask)
    if len(np.unique(label_img)) > num_unique_labels:
        regions = measure.regionprops(label_img)
        mask = mask.astype(bool)
        all_area = [i.area for i in regions]
        second_max = max([i for i in all_area if i != max(all_area)])
        mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
    return mask.astype(np.uint8)


def align(path_moving_image, path_fixed_image, show_images=False):
    # https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/10-wsi-registration.html
    mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
    mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode

    ON_GPU = False  # Should be changed to False if no cuda-enabled GPU is available

    warnings.filterwarnings("ignore")
    global_save_dir = Path("./tmp/")

    rmdir(global_save_dir)  # remove  directory if it exists from previous runs
    global_save_dir.mkdir()
    logger.info("Creating new directory %s", global_save_dir)

    fixed_img_file_name = Path(path_fixed_image)  # Change the file extension to .mrxs
    moving_img_file_name = Path(path_moving_image)  # Change the file extension to .mrxs

    # Assuming you have your MRXS files stored on your local PC, update the file paths accordingly.

    # For fixed image
    fixed_wsi_reader = WSIReader.open(fixed_img_file_name)
    fixed_image_rgb = fixed_wsi_reader.slide_thumbnail(resolution=0.1563, units="power")


    # For moving image
    moving_wsi_reader = WSIReader.open(moving_img_file_name)
    moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=0.1563, units="power")

    if show_images:
        _, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(fixed_image_rgb, cmap="gray")
        axs[0].set_title("Fixed Image")
        axs[1].imshow(moving_image_rgb, cmap="gray")
        axs[1].set_title("Moving Image")
        plt.show()

    # Preprocessing fixed and moving images
    fixed_image = preprocess_image(fixed_image_rgb)
    moving_image = preprocess_image(moving_image_rgb)
    fixed_image, moving_image = match_histograms(fixed_image, moving_image)

    # Visualising the results
    if show_images:
        _, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(fixed_image, cmap="gray")
        axs[0].set_title("Fixed Image")
        axs[1].imshow(moving_image, cmap="gray")
        axs[1].set_title("Moving Image")
        plt.show()

    temp = np.repeat(np.expand_dims(fixed_image, axis=2), 3, axis=2)
    _saved = cv2.imwrite(str(global_save_dir / "fixed.png"), temp)
    temp = np.repeat(np.expand_dims(moving_image, axis=2), 3, axis=2)
    _saved = cv2.imwrite(str(global_save_dir / "moving.png"), temp)

    save_dir = global_save_dir / "tissue_mask"
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=False, onerror=None)

    segmentor = SemanticSegmentor(
        pretrained_model="unet_tissue_mask_tsef",
        num_loader_workers=4,
        batch_size=4,
    )

    output = segmentor.predict(
        [
            global_save_dir / "fixed.png",
            global_save_dir / "moving.png",
        ],
        save_dir=save_dir,
        mode="tile",
        resolution=1.0,
        units="baseline",
        patch_input_shape=(1024, 1024),
        patch_output_shape=(512, 512),
        stride_shape=(512, 512),
        on_gpu=ON_GPU,
        crash_on_exception=False,
    )

    fixed_mask = np.load(output[0][1] + ".raw.0.npy")
    moving_mask = np.load(output[1][1] + ".raw.0.npy")

    # num of unique objects for segmentation is 2.
    num_unique_labels = 2

    # Simple processing of the raw prediction to generate semantic segmentation task
    fixed_mask = np.argmax(fixed_mask, axis=-1) == num_unique_labels
    moving_mask = np.argmax(moving_mask, axis=-1) == num_unique_labels

    fixed_mask = post_processing_mask(fixed_mask)
    moving_mask = post_processing_mask(moving_mask)

    if show_images:
        _, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(fixed_mask, cmap="gray")
        axs[0].set_title("Fixed Mask")
        axs[1].imshow(moving_mask, cmap="gray")
        axs[1].set_title("Moving Mask")
        plt.show()

    dfbr_fixed_image = np.repeat(np.expand_dims(fixed_image, axis=2), 3, axis=2)
    dfbr_moving_image = np.repeat(np.expand_dims(moving_image, axis=2), 3, axis=2)

    dfbr = DFBRegister()
    dfbr_transform = dfbr.register(
        dfbr_fixed_image,
        dfbr_moving_image,
        fixed_mask,
        moving_mask,
    )

    # Visualization
    original_moving = cv2.warpAffine(
        moving_image,
        np.eye(2, 3),
        fixed_image.shape[:2][::-1],
    )
    dfbr_registered_image = cv2.warpAffine(
        moving_image,
        dfbr_transform[0:-1],
        fixed_image.shape[:2][::-1],
    )

    before_overlay = np.dstack((original_moving, fixed_image, original_moving))
    dfbr_overlay = np.dstack((dfbr_registered_image, fixed_image, dfbr_registered_image))

    if show_images:
        _, axs = plt.subplots(1, 2, figsize=(15, 10))
        axs[0].imshow(before_overlay, cmap="gray")
        axs[0].set_title("Overlay Before Registration")
        axs[1].imshow(dfbr_overlay, cmap="gray")
        axs[1].set_title("Overlay After DFBR")
        plt.show()

    return dfbr_transform, fixed_wsi_reader, moving_wsi_reader

def normalize_image(image):
    # Convert image to float32 if not already
    image = image.astype(np.float32)
    # Normalize pixel values to range [0, 1]
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_normalized

def store_normalized_aligned_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader, moving_wsi_reader, size, tile_dir_fixed, tile_dir_moving):
    # https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/10-wsi-registration.html

    slide = load_image(path_fixed_image)
    level = 5
    _, [ymin, ymax, xmin, xmax] = crop_image(slide, level)  # at level 5 since lower memory-load
    xmin = xmin * ((level+1) ** 2)
    xmax = xmax * ((level+1) ** 2)
    ymin = ymin * ((level+1) ** 2)
    ymax = ymax * ((level+1) ** 2)

    # xmax, ymax = slide.level_dimensions[0]

    # DFBR transform is computed for level 7
    # Hence it should be mapped to level 0 for AffineWSITransformer
    dfbr_transform_level = 7
    transform_level0 = dfbr_transform * [
        [1, 1, 2 ** dfbr_transform_level],
        [1, 1, 2 ** dfbr_transform_level],
        [1, 1, 1],
    ]

    # Extract transformed region from the moving whole slide image
    tfm = AffineWSITransformer(moving_wsi_reader, transform_level0)

    tile_names = []
    print(xmax,ymax)
    for x in range(xmin, xmax, size[0]):
        for y in range(ymin, ymax, size[1]):
            location = (x, y)  # at base level 0

            # Extract region from the fixed whole slide image
            fixed_tile = fixed_wsi_reader.read_rect(location, size, resolution=20,
                                                    units="power")  # resolution at 20xzoom
            moving_tile = tfm.read_rect(location, size, resolution=20, units="power")  # resolution at 20xzoom

            if np.sum(np.any(fixed_tile != np.array([255, 255, 255]), axis=-1)) > (
                    fixed_tile.shape[0] * fixed_tile.shape[
                1] / 2):  # only safe tile if more than half of it is not white
                tile_name_fixed = os.path.join(tile_dir_fixed, os.path.split(tile_dir_fixed)[1] +  '%d_%d' % (x, y))
                print("Now saving tile with title: ", tile_name_fixed)
                fixed_tile = normalize_image(fixed_tile)
                plt.imsave(tile_name_fixed + ".png", fixed_tile)
                tile_name_moving = os.path.join(tile_dir_moving, os.path.split(tile_dir_moving)[1] +  '%d_%d' % (x, y))
                print("Now saving tile with title: ", tile_name_moving)
                moving_tile = normalize_image(moving_tile)
                plt.imsave(tile_name_moving + ".png", moving_tile)
                tile_names.append(fr'{tile_dir_fixed}\{x}_{y}.png')
    return tile_names


def KMeans_image(tile, show_images = False):
    # CHATGPT
    from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

    # Initial mean colors in RGB format (purple, yellow, green, brown, blue and white)
    # initial_means = np.array([[128, 0, 128], [255, 255, 0], [0, 128, 0], [165, 42, 42], [0, 0, 255],[255,255,255]])
    initial_means = np.array(
        [[198, 135, 255], [225, 247, 174], [0, 56, 44], [164, 101, 32], [146, 211, 255], [255, 255, 255]])

    # Load your image data (replace 'your_image_path' with the actual path to your image)
    image_data = tile

    # Reshape the image data to a 2D array where each row is a pixel's RGB values
    data = image_data.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=len(initial_means), init=initial_means, n_init=1, random_state=42)
    kmeans.fit(data)

    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the original image
    ax1.imshow(image_data)
    ax1.axis('off')
    ax1.set_title('Original Image')

    # Plot the segmented image
    ax2.imshow(labels.reshape(image_data.shape[:2]), cmap=ListedColormap(cluster_centers/255))
    ax2.axis('off')
    ax2.set_title('Segmented Image')

    plt.show()

    # 3D Visualization of K-means clusters
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    #IF YOU WANT EVERY DOT IN COLOR OF ASSIGNED CLUSTER
    # Scatter plot for each cluster
    for i in range(len(initial_means)):
        cluster_points = data[labels == i]
        ax_3d.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Label {i + 1}',
                      c=[cluster_centers[i] / 255.0])

    #IF YOU WANT EVERY DOT IN ORIGINAL COLOR
    #Scatter plot for each cluster
    # for i in range(len(initial_means)):
    #     cluster_points = data[labels == i]
    #     for point in (cluster_points):
    #         ax_3d.scatter(point[0], point[1], point[2], c=[point / 255.0])

    # Plot the initial means
    # ax_3d.scatter(initial_means[:, 0], initial_means[:, 1], initial_means[:, 2], c='black', marker='X', s=100,
    #             label='Initial Means')

    # Plot the final cluster centers
    # ax_3d.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='o', s=200,
    #             label='Final Cluster Centers')

    ax_3d.set_xlabel('Red')
    ax_3d.set_ylabel('Green')
    ax_3d.set_zlabel('Blue')
    ax_3d.legend()
    ax_3d.set_title('K-means Clustering in RGB Space')

    plt.show()
    return data, labels, cluster_centers


def KNN(tile, data, labels, cluster_centers, K=5, show_images=False):
    reference_labels = labels

    # Train k-NN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=K)
    knn_classifier.fit(data, reference_labels)

    # Reshape the new image data to a 2D array where each row is a pixel's RGB values
    input_tile = tile.reshape((-1, 3))

    # Segment the new image using the trained k-NN classifier
    predicted_labels = knn_classifier.predict(input_tile)

    # Reshape the predicted labels to the shape of the new image
    segmented_image = predicted_labels.reshape(tile.shape[:2])

    if show_images:
        # Plot the original and segmented images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the original image
        ax1.imshow(tile)
        ax1.axis('off')
        ax1.set_title('Original Image')

        # Plot the segmented image
        ax2.imshow(segmented_image, cmap=ListedColormap(cluster_centers / 255))
        ax2.axis('off')
        ax2.set_title('Segmented Image')
        plt.show()

    return predicted_labels


def soft_labeling(tile_names, data, labels, cluster_centers,tile_dir_moving):
    soft_labels = []
    for image in tile_names:
        tile_name_moving = os.path.join(tile_dir_moving, image)
        tile = np.array(Image.open(tile_name_moving).convert("RGB"))
        predicted_labels = KNN(tile, data, labels, cluster_centers)
        length = predicted_labels.shape[0]
        zeros = np.count_nonzero(predicted_labels == 0) / length
        ones = np.count_nonzero(predicted_labels == 1) / length
        twos = np.count_nonzero(predicted_labels == 2) / length
        threes = np.count_nonzero(predicted_labels == 3) / length
        fours = np.count_nonzero(predicted_labels == 4) / length
        fives = np.count_nonzero(predicted_labels == 5) / length
        label = [zeros, ones, twos, threes, fours, fives]
        soft_labels.append(label)
    return soft_labels


def create_dataset(tile_names, soft_labels, location_data_csv):
    #data = zip(tile_names, soft_labels)
    data = zip(tile_names,*zip(*soft_labels))
    # Write to CSV file
    with open(location_data_csv, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow(['image','label0','label1','label2','label3','label4','label5'])

        # Write the data
        csv_writer.writerows(data)
