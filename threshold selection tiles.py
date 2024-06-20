import os
import json
from functions import load_image,crop_image,enough_filled_no_norm,enough_entropy,variance_of_laplacian
from tiatoolbox.wsicore.wsireader import WSIReader
import numpy as np
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':

    config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote_new/configuration.json"
    with open(config_filename, 'r') as config_json:
        config = json.load(config_json)

    path_biopsies = config["other_path_biopsies"]
    tile_path_init = config["tile_path_extra_filtered"]
    size = config["variables"]["size_image"] #
    size = (int(size[0]/0.242534722222222), int(size[1]/0.242647058823529)) #convert micrometers to pixels (in 20x zoom)

    biopsy_sets = os.listdir(path_biopsies)

    cmap = plt.get_cmap('viridis')
    # Sample 12 evenly spaced values from the colormap
    num_colors = 12
    colors = [cmap(i / (num_colors - 1))[:3] for i in range(num_colors)]  # Extract RGB values, ignoring alpha
    # Convert RGB values from 0-1 range to 0-255 range
    colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    # Category names corresponding to each color
    category_names = ["<55","55-70","70-85","85-100","100-115","115-130","130-145","145-160","160-175","175-190","190-205",">205"]



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

        fixed_crop[np.all(fixed_crop == [0, 0, 0], axis=-1)] = [255, 255, 255]
        plt.figure()
        plt.imshow(fixed_crop)
        plt.axis('off')
        plt.show()

        # for x_fixed in range(int(xmin_fixedo),
        #                      int(xmax_fixedo), size[0]):
        #     for y_fixed in range(int(ymin_fixedo),
        #                          int(ymax_fixedo), size[1]):
        #         location_fixed = (x_fixed, y_fixed)  # at base level 0
        #         fixed_tile = np.array(slide_fixed.read_region(location_fixed, 0, size))  # resolution at 20xzoom
        #         if enough_filled_no_norm(fixed_tile, 0.25, 0.3):
        #             x = (x_fixed - xmin_fixedo) / 32
        #             y = (y_fixed - ymin_fixedo) / 32
        #             image = cv2.cvtColor(fixed_tile, cv2.COLOR_RGB2GRAY)
        #             variance = cv2.Laplacian(image, cv2.CV_64F).var()
        #             if variance < 55:
        #                 color = colors_255[0]
        #             elif variance < 70:
        #                 color = colors_255[1]
        #             elif variance < 85:
        #                 color = colors_255[2]
        #             elif variance < 100:
        #                 color = colors_255[3]
        #             elif variance < 115:
        #                 color = colors_255[4]
        #             elif variance < 130:
        #                 color = colors_255[5]
        #             elif variance < 145:
        #                 color = colors_255[6]
        #             elif variance < 160:
        #                 color = colors_255[7]
        #             elif variance < 175:
        #                 color = colors_255[8]
        #             elif variance < 190:
        #                 color = colors_255[9]
        #             elif variance < 205:
        #                 color = colors_255[10]
        #             else:
        #                 color = colors_255[11]
        #
        #             fixed_crop[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = color
        #     print(f'x: {x_fixed} / {int(xmax_fixedo)}')
        # fixed_crop[np.all(fixed_crop == [0, 0, 0], axis=-1)] = [255, 255, 255]
        # plt.figure()
        # plt.imshow(fixed_crop)
        # plt.title('threshold Laplacian: 110')
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
        # ax[0].imshow(fixed_crop)
        # ax[0].axis('off')
        # for idx, (color, category_name) in enumerate(zip(colors_255, category_names)):
        #     ax[1].add_patch(plt.Rectangle((0, idx), 1, 1, color=np.array(color) / 255.0))
        #     ax[1].text(1.1, idx + 0.5, category_name, color='black', va='center', fontsize=12)
        # # Set plot limits and remove axes
        # ax[1].set_xlim(0, 2)
        # ax[1].set_ylim(0, num_colors)
        # ax[1].axis('off')
        # ax[1].set_title('Variance of Laplacian', fontsize=16, pad=20)
        # # Show the plot
        # plt.tight_layout()
        # plt.show()

        moving_crop[np.all(moving_crop == [0, 0, 0], axis=-1)] = [255, 255, 255]
        plt.figure()
        plt.imshow(moving_crop)
        plt.axis('off')
        plt.show()

        for x_moving in range(int(xmin_movingo),
                              int(xmax_movingo), size[0]):
            for y_moving in range(int(ymin_movingo),
                                  int(ymax_movingo), size[1]):
                location_moving = (x_moving, y_moving)  # at base level 0
                # Extract region from the fixed whole slide image
                moving_tile = np.array(slide_moving.read_region(location_moving, 0, size))  # resolution at 20xzoom
                if enough_filled_no_norm(moving_tile, 0.25, 0.3):
                    x = (x_moving - xmin_movingo) / 32
                    y = (y_moving - ymin_movingo) / 32
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
                    moving_crop[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = color
            print(f'x: {x_moving} / {int(xmax_movingo)}')
        moving_crop[np.all(moving_crop == [0, 0, 0], axis=-1)] = [255, 255, 255]
        plt.figure()
        plt.imshow(moving_crop)
        plt.title('threshold Laplacian: 90')
        plt.show()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(moving_crop)
        ax[0].axis('off')
        for idx, (color, category_name) in enumerate(zip(colors_255, category_names)):
            ax[1].add_patch(plt.Rectangle((0, idx), 1, 1, color=np.array(color) / 255.0))
            ax[1].text(1.1, idx + 0.5, category_name, color='black', va='center', fontsize=12)
        # Set plot limits and remove axes
        ax[1].set_xlim(0, 2)
        ax[1].set_ylim(0, num_colors)
        ax[1].axis('off')
        ax[1].set_title('Variance of Laplacian', fontsize=16, pad=20)

        # Show the plot
        plt.tight_layout()
        plt.show()
        print("Done visualization")
        return

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

        store_tiles(path_fixed_image, path_moving_image, size, tile_dir_fixed, tile_dir_moving, locs_fixed,
                    locs_moving)


    for biopsy_set in biopsy_sets:
        if biopsy_set == 'Janssen':
            path_biopsy_set = os.path.join(path_biopsies, biopsy_set)
            tile_path = os.path.join(tile_path_init, biopsy_set)
            patient_biopsies = os.listdir(path_biopsy_set)
            patient_biopsies_fixed = sorted([biopsy for biopsy in patient_biopsies if 'HE' in biopsy and biopsy.lower().endswith('.mrxs')])
            patient_biopsies_moving = sorted([biopsy for biopsy in patient_biopsies if 'MUC' in biopsy and biopsy.lower().endswith('.mrxs')])

            patients_fixed = [p.replace('HE', '') for p in patient_biopsies_fixed]
            patients_moving = [p.replace('CDX2p_MUC2y_MUC5g_CD8dab', '') for p in patient_biopsies_moving]
            patient_biopsies_fixed = sorted([p for p in patient_biopsies_fixed if p.replace('HE', '') in patients_moving])
            patient_biopsies_moving = sorted([p for p in patient_biopsies_moving if p.replace('CDX2p_MUC2y_MUC5g_CD8dab', '') in patients_fixed])
            patient_biopsy_moving = 'B-1986096_B3_CDX2p_MUC2y_MUC5g_CD8dab.mrxs'
            patient_biopsy_fixed = 'B-1986096_B3_HE.mrxs'
            process_patient_biopsy(patient_biopsy_fixed, patient_biopsy_moving, path_biopsy_set, tile_path)