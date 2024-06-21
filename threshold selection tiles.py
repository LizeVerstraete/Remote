import os
import json
from functions import load_image, crop_image, enough_filled_no_norm, enough_entropy, variance_of_laplacian
from tiatoolbox.wsicore.wsireader import WSIReader
import numpy as np
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':

    path_biopsies = "/esat/smcdata/tempusers/r0786880/no_backup/Biopsies"
    tile_path_init = "/esat/biomeddata/kkontras/r0786880/biopsy_data_bigger_dataset_412_entropy_norm11"
    size = [100,100] #
    size = (
    int(size[0] / 0.242534722222222), int(size[1] / 0.242647058823529))  # convert micrometers to pixels (in 20x zoom)

    biopsy_sets = os.listdir(path_biopsies)

    cmap = plt.get_cmap('viridis')
    # Sample 12 evenly spaced values from the colormap
    num_colors = 12
    colors = [cmap(i / (num_colors - 1))[:3] for i in range(num_colors)]  # Extract RGB values, ignoring alpha
    # Convert RGB values from 0-1 range to 0-255 range
    colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    # Category names corresponding to each color
    category_names = ["<55", "55-70", "70-85", "85-100", "100-115", "115-130", "130-145", "145-160", "160-175",
                      "175-190", "190-205", ">205"]


    def store_tiles(path_fixed_image, path_moving_image, size, tile_dir_fixed, tile_dir_moving, locs_fixed,
                    locs_moving):
        slide_fixed = load_image(path_fixed_image)
        level = 5
        fixed_crop, [ymin_fixedo, ymax_fixedo, xmin_fixedo, xmax_fixedo] = crop_image(slide_fixed,
                                                                                      level)  # at level 5 since lower memory-load
        xmin_fixedo = xmin_fixedo * (2 ** (level))
        xmax_fixedo = xmax_fixedo * (2 ** (level))
        ymin_fixedo = ymin_fixedo * (2 ** (level))
        ymax_fixedo = ymax_fixedo * (2 ** (level))

        slide_moving = load_image(path_moving_image)
        level = 5
        moving_crop, [ymin_movingo, ymax_movingo, xmin_movingo, xmax_movingo] = crop_image(slide_moving,
                                                                                           level)  # at level 5 since lower memory-load
        xmin_movingo = xmin_movingo * (2 ** (level))
        xmax_movingo = xmax_movingo * (2 ** (level))
        ymin_movingo = ymin_movingo * (2 ** (level))
        ymax_movingo = ymax_movingo * (2 ** (level))

        no_background_fixed = fixed_crop.copy()
        laplacian_fixed = fixed_crop.copy()
        end_result_fixed = fixed_crop.copy()
        for x_fixed in range(int(xmin_fixedo),
                             int(xmax_fixedo), size[0]):
            for y_fixed in range(int(ymin_fixedo),
                                 int(ymax_fixedo), size[1]):
                location_fixed = (x_fixed, y_fixed)  # at base level 0
                # Extract region from the fixed whole slide image
                fixed_tile = np.array(slide_fixed.read_region(location_fixed, 0, size))  # resolution at 20xzoom
                x = (x_fixed - xmin_fixedo) / 32
                y = (y_fixed - ymin_fixedo) / 32
                if enough_filled_no_norm(fixed_tile, 0.25, 0.3):
                    image = cv2.cvtColor(fixed_tile, cv2.COLOR_RGB2GRAY)
                    variance = cv2.Laplacian(image, cv2.CV_64F).var()
                    if variance < 55:
                        color = colors_255[0]
                    elif variance < 70:
                        color = colors_255[1]
                    elif variance < 85:
                        color = colors_255[2]
                    elif variance < 100:
                        color = colors_255[3]
                    elif variance < 115:
                        color = colors_255[4]
                    elif variance < 130:
                        color = colors_255[5]
                    elif variance < 145:
                        color = colors_255[6]
                    elif variance < 160:
                        color = colors_255[7]
                    elif variance < 175:
                        color = colors_255[8]
                    elif variance < 190:
                        color = colors_255[9]
                    elif variance < 205:
                        color = colors_255[10]
                    else:
                        color = colors_255[11]
                    laplacian_fixed[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = color
                    if variance < 100:
                        end_result_fixed[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                else:
                    laplacian_fixed[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                    no_background_fixed[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                    end_result_fixed[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
            print(f'x: {x_fixed} / {int(xmax_fixedo)}')

        no_background = moving_crop.copy()
        laplacian = moving_crop.copy()
        end_result = moving_crop.copy()
        for x_moving in range(int(xmin_movingo),
                              int(xmax_movingo), size[0]):
            for y_moving in range(int(ymin_movingo),
                                  int(ymax_movingo), size[1]):
                location_moving = (x_moving, y_moving)  # at base level 0
                # Extract region from the fixed whole slide image
                moving_tile = np.array(slide_moving.read_region(location_moving, 0, size))  # resolution at 20xzoom
                x = (x_moving - xmin_movingo) / 32
                y = (y_moving - ymin_movingo) / 32
                if enough_filled_no_norm(moving_tile, 0.25, 0.3):
                    image = cv2.cvtColor(moving_tile, cv2.COLOR_RGB2GRAY)
                    variance = cv2.Laplacian(image, cv2.CV_64F).var()
                    if variance < 55:
                        color = colors_255[0]
                    elif variance < 70:
                        color = colors_255[1]
                    elif variance < 85:
                        color = colors_255[2]
                    elif variance < 100:
                        color = colors_255[3]
                    elif variance < 115:
                        color = colors_255[4]
                    elif variance < 130:
                        color = colors_255[5]
                    elif variance < 145:
                        color = colors_255[6]
                    elif variance < 160:
                        color = colors_255[7]
                    elif variance < 175:
                        color = colors_255[8]
                    elif variance < 190:
                        color = colors_255[9]
                    elif variance < 205:
                        color = colors_255[10]
                    else:
                        color = colors_255[11]
                    laplacian[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = color
                    if variance < 100:
                        end_result[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                else:
                    laplacian[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                    no_background[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
                    end_result[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 0]
            print(f'x: {x_moving} / {int(xmax_movingo)}')

        return moving_crop,fixed_crop,no_background,no_background_fixed,laplacian,laplacian_fixed,end_result,end_result_fixed


    def process_patient_biopsy(patient_biopsy_fixed, patient_biopsy_moving, path_biopsy_set, tile_path):
        locations = {
            "B-1986096_B3_HE": [15773, 48738, 49140, 67608],
            "B-1986096_B3_CDX2p_MUC2y_MUC5g_CD8dab": [18000, 52683, 49140, 67608]
        }
        print('ready to process ', patient_biopsy_fixed)
        key_moving = patient_biopsy_moving.replace('.mrxs', '')
        key_fixed = patient_biopsy_fixed.replace('.mrxs', '')
        locs_fixed = locations[key_fixed]
        locs_moving = locations[key_moving]
        path_fixed_image = os.path.join(path_biopsy_set, patient_biopsy_fixed)
        path_moving_image = os.path.join(path_biopsy_set, patient_biopsy_moving)
        tile_dir_fixed = os.path.join(tile_path, patient_biopsy_fixed.replace('.mrxs', ''))
        tile_dir_moving = os.path.join(tile_path, patient_biopsy_moving.replace('.mrxs', ''))

        moving_crop,fixed_crop,no_background,no_background_fixed,laplacian,laplacian_fixed,end_result,end_result_fixed = store_tiles(path_fixed_image, path_moving_image, size, tile_dir_fixed, tile_dir_moving, locs_fixed,
                    locs_moving)
        return moving_crop,fixed_crop,no_background,no_background_fixed,laplacian,laplacian_fixed,end_result,end_result_fixed


    for biopsy_set in biopsy_sets:
        if biopsy_set == 'Janssen':
            path_biopsy_set = os.path.join(path_biopsies, biopsy_set)
            tile_path = os.path.join(tile_path_init, biopsy_set)
            patient_biopsies = os.listdir(path_biopsy_set)
            patient_biopsies_fixed = sorted(
                [biopsy for biopsy in patient_biopsies if 'HE' in biopsy and biopsy.lower().endswith('.mrxs')])
            patient_biopsies_moving = sorted(
                [biopsy for biopsy in patient_biopsies if 'MUC' in biopsy and biopsy.lower().endswith('.mrxs')])

            patients_fixed = [p.replace('HE', '') for p in patient_biopsies_fixed]
            patients_moving = [p.replace('CDX2p_MUC2y_MUC5g_CD8dab', '') for p in patient_biopsies_moving]
            patient_biopsies_fixed = sorted(
                [p for p in patient_biopsies_fixed if p.replace('HE', '') in patients_moving])
            patient_biopsies_moving = sorted(
                [p for p in patient_biopsies_moving if p.replace('CDX2p_MUC2y_MUC5g_CD8dab', '') in patients_fixed])
            patient_biopsy_moving = 'B-1986096_B3_CDX2p_MUC2y_MUC5g_CD8dab.mrxs'
            patient_biopsy_fixed = 'B-1986096_B3_HE.mrxs'
            moving_crop,fixed_crop,no_backgr_moving,no_backgr_fixed,laplacian_moving,laplacian_fixed,end_result_moving,end_result_fixed = process_patient_biopsy(patient_biopsy_fixed, patient_biopsy_moving, path_biopsy_set, tile_path)

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(fixed_crop)
            plt.axis('off')
            plt.title("WSI H&E")
            plt.show()

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(no_backgr_fixed)
            plt.axis('off')
            plt.title("remove background")
            plt.show()

            # Assuming laplacian_fixed, colors_255, and category_names are defined
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
            # Display the image on the first subplot
            ax[0].imshow(laplacian_fixed)
            ax[0].axis('off')
            # Create a color legend on the second subplot
            for idx, (color, category_name) in enumerate(zip(colors_255, category_names)):
                ax[1].add_patch(plt.Rectangle((0, idx), 1, 1, color=np.array(color) / 255.0))
                ax[1].text(1.1, idx + 0.5, category_name, color='black', va='center', fontsize=12)
            # Set plot limits and remove axes for the second subplot
            ax[1].set_xlim(0, 2)
            ax[1].set_ylim(0, len(colors_255))
            ax[1].axis('off')
            # Set the overall figure title
            fig.suptitle("Variance of Laplacian")
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=-0.4)  # Decrease this value to reduce the space between subplots
            # Display the plot
            plt.show()

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(end_result_fixed)
            plt.axis('off')
            plt.title("final selection of tiles")
            plt.show()

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(moving_crop)
            plt.axis('off')
            plt.title("WSI IHC")
            plt.show()

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(no_backgr_moving)
            plt.axis('off')
            plt.title("remove background")
            plt.show()

            # Assuming laplacian_fixed, colors_255, and category_names are defined
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
            # Display the image on the first subplot
            ax[0].imshow(laplacian_moving)
            ax[0].axis('off')
            # Create a color legend on the second subplot
            for idx, (color, category_name) in enumerate(zip(colors_255, category_names)):
                ax[1].add_patch(plt.Rectangle((0, idx), 1, 1, color=np.array(color) / 255.0))
                ax[1].text(1.1, idx + 0.5, category_name, color='black', va='center', fontsize=12)
            # Set plot limits and remove axes for the second subplot
            ax[1].set_xlim(0, 2)
            ax[1].set_ylim(0, len(colors_255))
            ax[1].axis('off')
            # Set the overall figure title
            fig.suptitle("Variance of Laplacian")
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=-0.4)  # Decrease this value to reduce the space between subplots
            # Display the plot
            plt.show()

            plt.figure()
            plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            plt.imshow(end_result_moving)
            plt.axis('off')
            plt.title("final selection of tiles")
            plt.show()

            fig, ax = plt.subplots(2, 4, figsize=(20, 10))
            # First row of images
            ax[0, 0].imshow(fixed_crop)
            ax[0, 0].set_title("WSI H&E")
            ax[0, 0].axis('off')
            ax[0, 1].imshow(no_backgr_fixed)
            ax[0, 1].set_title("remove background")
            ax[0, 1].axis('off')
            # Display the image on the third subplot in the first row
            ax[0, 2].imshow(laplacian_fixed)
            ax[0, 2].axis('off')
            ax[0, 2].set_title("Variance of Laplacian")
            # Display the image on the third subplot in the first row
            ax[0, 3].imshow(end_result_fixed)
            ax[0, 3].axis('off')
            ax[0, 3].set_title("final selection of tiles")
            ax[1, 0].imshow(moving_crop)
            ax[1, 0].set_title("WSI IHC")
            ax[1, 0].axis('off')
            ax[1, 1].imshow(no_backgr_moving)
            ax[1, 1].set_title("remove background")
            ax[1, 1].axis('off')
            # Display the image on the third subplot in the second row
            ax[1, 2].imshow(laplacian_moving)
            ax[1, 2].axis('off')
            ax[1, 2].set_title("Variance of Laplacian")
            ax[1, 3].imshow(end_result_moving)
            ax[1, 3].axis('off')
            ax[1, 3].set_title("final selection of tiles")
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust as needed to reduce/increase space between subplots
            # Display the plot
            plt.show()
