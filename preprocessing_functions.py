import os
import openslide
import logging
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
import math
import numpy as np
from matplotlib import pyplot as plt

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

    return cropped_image, location

def normalize_image(image):
    # Convert image to float32 if not already
    image = image.astype(np.float32)
    image_normalized = (image - np.mean(image)) / np.std(image)
    return image_normalized

def get_image_dimensions(path_fixed_image, path_moving_image):
    slide_fixed = load_image(path_fixed_image)
    slide_moving = load_image(path_moving_image)
    level = 5
    slide_np = np.array(slide_fixed.read_region((0, 0), level, slide_fixed.level_dimensions[level]).convert('RGB')).shape
    slide_np2 = np.array(slide_moving.read_region((0, 0), level, slide_moving.level_dimensions[level]).convert('RGB')).shape
    slide_np = tuple(dim * ((level + 1) ** 2) for dim in slide_np)
    slide_np2 = tuple(dim * ((level + 1) ** 2) for dim in slide_np2)
    _, [ymin_fixed, ymax_fixed, xmin_fixed, xmax_fixed] = crop_image(slide_fixed, level)  # at level 5 since lower memory-load
    xmin_fixed = xmin_fixed * ((level+1) ** 2)
    xmax_fixed = xmax_fixed * ((level+1) ** 2)
    ymin_fixed = ymin_fixed * ((level+1) ** 2)
    ymax_fixed = ymax_fixed * ((level+1) ** 2)
    print("dimensions whole fixed slide", slide_np)
    print("dimensions cropping", xmin_fixed,xmax_fixed,ymin_fixed,ymax_fixed)
    print("dimensions whole moving slide", slide_np2)
    return

def store_normalized_tiles(path_fixed_image, path_moving_image, size, tile_dir_fixed, tile_dir_moving):
    # https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/10-wsi-registration.html

    slide_fixed = load_image(path_fixed_image)
    level = 5
    _, [ymin_fixed, ymax_fixed, xmin_fixed, xmax_fixed] = crop_image(slide_fixed, level)  # at level 5 since lower memory-load
    xmin_fixed = xmin_fixed * ((level+1) ** 2)
    xmax_fixed = xmax_fixed * ((level+1) ** 2)
    ymin_fixed = ymin_fixed * ((level+1) ** 2)
    ymax_fixed = ymax_fixed * ((level+1) ** 2)

    slide_moving = load_image(path_moving_image)
    level = 5
    _, [ymin_moving, ymax_moving, xmin_moving, xmax_moving] = crop_image(slide_moving, level)  # at level 5 since lower memory-load
    xmin_moving = xmin_moving * ((level+1) ** 2)
    xmax_moving = xmax_moving * ((level+1) ** 2)
    ymin_moving = ymin_moving * ((level+1) ** 2)
    ymax_moving = ymax_moving * ((level+1) ** 2)

    print(xmax_fixed,ymax_fixed)
    print(xmax_moving,ymax_moving)
    for x_moving in range(xmin_moving, xmax_moving, size[0]):
        for y_moving in range(ymin_moving, ymax_moving, size[1]):
            location_moving = (x_moving, y_moving)  # at base level 0
            # Extract region from the fixed whole slide image
            moving_tile = slide_moving.read_region(location_moving, 0, size)  # resolution at 20xzoom
            moving_tile = normalize_image(np.array(moving_tile))
            if enough_filled(moving_tile, 0.25, 0.3):
                tile_name_moving = os.path.join(tile_dir_moving, os.path.split(tile_dir_moving)[1] +  '%d_%d' % (x_moving, y_moving))
                print("Now saving tile with title: ", tile_name_moving, "")
                plt.imsave(tile_name_moving + ".png", moving_tile)
    print("Done saving moving tiles",tile_dir_moving)
    for x_fixed in range(xmin_fixed, xmax_fixed, size[0]):
        for y_fixed in range(ymin_fixed, ymax_fixed, size[1]):
            location_fixed = (x_fixed, y_fixed)  # at base level 0

            fixed_tile = slide_fixed.read_region(location_fixed, 0, size)  # resolution at 20xzoom
            fixed_tile = normalize_image(np.array(fixed_tile))

            if enough_filled(fixed_tile, 0.25, 0.3):
                tile_name_fixed = os.path.join(tile_dir_fixed, os.path.split(tile_dir_fixed)[1] +  '%d_%d' % (x_fixed, y_fixed))
                print("Now saving tile with title: ", tile_name_fixed)
                plt.imsave(tile_name_fixed + ".png", fixed_tile)
    print("Done saving fixed tiles",tile_dir_fixed)
    return

def show_normalized_tiles(path_fixed_image, path_moving_image, size):
    # https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/10-wsi-registration.html

    slide_fixed = load_image(path_fixed_image)
    level = 5
    _, [ymin_fixed, ymax_fixed, xmin_fixed, xmax_fixed] = crop_image(slide_fixed, level)  # at level 5 since lower memory-load
    xmin_fixed = xmin_fixed * ((level+1) ** 2)
    xmax_fixed = xmax_fixed * ((level+1) ** 2)
    ymin_fixed = ymin_fixed * ((level+1) ** 2)
    ymax_fixed = ymax_fixed * ((level+1) ** 2)

    slide_moving = load_image(path_moving_image)
    level = 5
    _, [ymin_moving, ymax_moving, xmin_moving, xmax_moving] = crop_image(slide_moving, level)  # at level 5 since lower memory-load
    xmin_moving = xmin_moving * ((level+1) ** 2)
    xmax_moving = xmax_moving * ((level+1) ** 2)
    ymin_moving = ymin_moving * ((level+1) ** 2)
    ymax_moving = ymax_moving * ((level+1) ** 2)

    print(xmax_fixed,ymax_fixed)
    print(xmax_moving,ymax_moving)
    for x_moving in range(xmin_moving, xmax_moving, size[0]):
        for y_moving in range(ymin_moving, ymax_moving, size[1]):
            location_moving = (x_moving, y_moving)  # at base level 0
            moving_tile = slide_moving.read_region(location_moving, 0, size)  # resolution at 20xzoom
            moving_tile = normalize_image(np.array(moving_tile))
            if enough_filled(moving_tile, 0.25, 0.3):
                plt.imshow(moving_tile)
                plt.title("save")
                plt.show()
            else:
                plt.imshow(moving_tile)
                plt.title("remove")
                plt.show()

    for x_fixed in range(xmin_fixed, xmax_fixed, size[0]):
        for y_fixed in range(ymin_fixed, ymax_fixed, size[1]):
            location_fixed = (x_fixed, y_fixed)
            fixed_tile = slide_fixed.read_region(location_fixed, 0, size)  # resolution at 20xzoom
            fixed_tile = normalize_image(np.array(fixed_tile))
            if enough_filled(fixed_tile, 0.25, 0.3):
                plt.imshow(fixed_tile)
                plt.title("save")
                plt.show()
            else:
                plt.imshow(fixed_tile)
                plt.title("remove")
                plt.show()
    return

def enough_filled(image,threshold,sample_freq):
    width = image.shape[0]
    height = image.shape[1]
    total = 0
    nb_none_white = 0
    for y in range(0,height,max(1, math.ceil(height * sample_freq))):
        for x in range(0,width,max(1, math.ceil(width * sample_freq))):
            total += 1
            if image[x][y][0] < 0.98 and image[x][y][1] < 0.98 and image[x][y][2] < 0.98:
                nb_none_white += 1
    return nb_none_white > total*threshold