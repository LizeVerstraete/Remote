import os
import json
from concurrent.futures import ThreadPoolExecutor
import functions

if __name__ == '__main__':

    config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote_new/configuration.json"
    with open(config_filename, 'r') as config_json:
        config = json.load(config_json)

    remove_existing_tiles = False
    path_biopsies = config["other_path_biopsies"]
    tile_path_init = config["tile_path_extra_filtered"]
    size = config["variables"]["size_image"] #
    size = (int(size[0]/0.242534722222222), int(size[1]/0.242647058823529)) #convert micrometers to pixels (in 20x zoom)

    biopsy_sets = os.listdir(path_biopsies)

    def process_patient_biopsy(patient_biopsy_fixed, patient_biopsy_moving, path_biopsy_set, tile_path):
        locations = {
            "B-1986096_B3_HE": [15773, 48738, 49140, 67608],
            "B-1986096_B3_CDX2p_MUC2y_MUC5g_CD8dab": [18000, 52683, 49140, 67608],
            "B-1986096_B4_HE": [25662, 52035, 3000, 24600],
            "B-1986096_B4_CDX2p_MUC2y_MUC5g_CD8dab": [12476, 42145, 6150, 30750],
            "B-1986096_B5_HE": [12476, 42145, 55269, 67608],
            "B-1986096_B5_CDX2p_MUC2y_MUC5g_CD8dab": [12476, 42145, 55284, 69144],
            "B-1986096_B6_HE": [22371, 40839, 6150, 24600],
            "B-1986096_B6_CDX2p_MUC2y_MUC5g_CD8dab": [23058, 39528, 6150, 24600],
            "B-1986183_B6_HE": [22366, 48738, 4614, 18456],
            "B-1986183_B6_CDX2p_MUC2y_MUC5g_CD8dab": [23672, 50055, 49140, 61452],
            "B-1986183_B7_HE": [15773, 45442, 61452, 70000],
            "B-1986183_B7_CDX2p_MUC2y_MUC5g_CD8dab": [13176, 42822, 12300, 24600],
            "B-1986183_B8_HE": [17094, 40839, 6150, 24600],
            "B-1986183_B8_CDX2p_MUC2y_MUC5g_CD8dab": [22366, 45442, 49140, 67608],
            "B-1986943_B6_HE": [7904, 51379, 61452, 70000],
            "B-1986943_B6_CDX2p_MUC2y_MUC5g_CD8dab": [7904, 51379, 18450, 26600],
            "B-1986943_B7_HE": [11856, 47427, 18450, 24600],
            "B-1986943_B7_CDX2p_MUC2y_MUC5g_CD8dab": [9882, 36234, 59904, 69144],
            "B-1986943_B8_HE": [25662, 45442, 61452, 67608],
            "B-1986943_B8_CDX2p_MUC2y_MUC5g_CD8dab": [38165, 59271, 18456, 26000],
            "B-1989502_B11_HE": [19069, 48738, 6150, 30750],
            "B-1989502_B11_CDX2p_MUC2y_MUC5g_CD8dab": [19069, 48738, 6150, 30750],
            "B-1989502_B3_HE": [9882, 336234, 4614, 23070],
            "B-1989502_B3_CDX2p_MUC2y_MUC5g_CD8dab": [7904, 43475, 5614, 23070],
            "B-1989502_B5_HE": [9882, 36234, 46000, 70000],
            "B-1989502_B5_CDX2p_MUC2y_MUC5g_CD8dab": [15773, 38849, 46000, 70000],
            "B-1989502_B6_HE": [23672, 50055, 4614, 23070],
            "B-1989502_B6_CDX2p_MUC2y_MUC5g_CD8dab": [15773, 42145, 4614, 23070],
            "B-1989502_B7_HE": [7904, 35570, 49140, 67608],
            "B-1989502_B7_CDX2p_MUC2y_MUC5g_CD8dab": [15773, 42145, 49140, 67608],
            "B-1989502_B8_HE": [19069, 45442, 3000, 24600],
            "B-1989502_B8_CDX2p_MUC2y_MUC5g_CD8dab": [19069, 45500, 3000, 24600],
            "B-1989502_B9_HE": [19069, 48738, 49140, 67608],
            "B-1989502_B9_CDX2p_MUC2y_MUC5g_CD8dab": [22366, 52035, 49140, 67608],
            "B-1993721_B5_HE": [23713, 47427, 6150, 30750],
            "B-1993721_B5_CDX2p_MUC2y_MUC5g_CD8dab": [25662, 48738, 6150, 30750],
            "B-1993721_B6_HE": [231044, 56638, 49140, 67608],
            "B-1993721_B6_CDX2p_MUC2y_MUC5g_CD8dab": [11818, 43477, 49140, 67608],
            "B-1993721_B7_HE": [15773, 52035, 6150, 30750],
            "B-1993721_B7_CDX2p_MUC2y_MUC5g_CD8dab": [6588, 42822, 4614, 27000],
            "B-1993721_B9_HE": [17089, 52683, 6150, 30750],
            "B-1993721_B9_CDX2p_MUC2y_MUC5g_CD8dab": [17089, 52683, 49140, 67608],
            "B-2027329_B1-10_HE": [15773, 52035, 6150, 18450],
            "B-2027329_B1-10_CDX2p_MUC2y_MUC5g_CD8dab": [22366, 52035, 49140, 61452],
            "B-2027329_B1-11_HE": [9882, 39528, 47000, 61452],
            "B-2027329_B1-11_CDX2p_MUC2y_MUC5g_CD8dab": [9882, 39528, 3000, 18450],
            "B-2027329_B1-4_HE": [15773, 45442, 49140, 67608],
            "B-2027329_B1-4_CDX2p_MUC2y_MUC5g_CD8dab": [11818, 43477, 49140, 67608],
            "B-2027329_B1-5_HE": [12476, 45442, 6150, 18450],
            "B-2027329_B1-5_CDX2p_MUC2y_MUC5g_CD8dab": [12476, 45442, 4614, 18450],
            "B-2027329_B1-6_HE": [22366, 52035, 49140, 67608],
            "B-2027329_B1-6_CDX2p_MUC2y_MUC5g_CD8dab": [13176, 42822, 49140, 67608],
            "B-2027329_B1-7_HE": [17089, 44773, 6150, 24600],
            "B-2027329_B1-7_CDX2p_MUC2y_MUC5g_CD8dab": [11818, 38201, 5150, 24600],
            "B-2027329_B1-8_HE": [9221, 36884, 49140, 67608],
            "B-2027329_B1-8_CDX2p_MUC2y_MUC5g_CD8dab": [15773, 45442, 49140, 67608],
            "B-2027329_B1-9_HE": [15809, 43475, 49140, 67608],
            "B-2027329_B1-9_CDX2p_MUC2y_MUC5g_CD8dab": [34878, 61251, 4000, 24600],
            "B-2028243_B1-04_HE": [11856, 43457, 49140, 67608],
            "B-2028243_B1-04_CDX2p_MUC2y_MUC5g_CD8dab": [11849, 43446, 3000, 24600]
        }
        try:
            print('ready to process ', patient_biopsy_fixed)
            key_moving = patient_biopsy_moving.replace('.mrxs', '')
            key_fixed = patient_biopsy_fixed.replace('.mrxs', '')
            locs_fixed = locations[key_fixed]
            locs_moving = locations[key_moving]
            path_fixed_image = os.path.join(path_biopsy_set, patient_biopsy_fixed)
            path_moving_image = os.path.join(path_biopsy_set, patient_biopsy_moving)
            tile_path_fixed = os.path.join(tile_path, patient_biopsy_fixed.replace('.mrxs', ''))
            tile_path_moving = os.path.join(tile_path, patient_biopsy_moving.replace('.mrxs', ''))
            if not os.path.exists(tile_path_fixed):
                os.makedirs(tile_path_fixed)
            if not os.path.exists(tile_path_moving):
                os.makedirs(tile_path_moving)
            if remove_existing_tiles:
                if os.listdir(tile_path_fixed):
                    for file in os.listdir(tile_path_fixed):
                        file_path = os.path.join(tile_path_fixed, file)
                        os.remove(file_path)
                if os.listdir(tile_path_moving):
                    for file in os.listdir(tile_path_moving):
                        file_path = os.path.join(tile_path_moving, file)
                        os.remove(file_path)
            #dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(path_moving_image, path_fixed_image,
            #                                                                      show_images=False)
            #functions.store_aligned_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader, moving_wsi_reader, size, None, None)
            functions.store_tiles(path_fixed_image,path_moving_image,size,tile_path_fixed,tile_path_moving,locs_fixed,locs_moving)
            print('finished processing ', patient_biopsy_fixed)
        except:
            print('didnt process ', patient_biopsy_fixed)
            return

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
            print('going to enter parallelism')
            with ThreadPoolExecutor(max_workers=80) as executor:
                executor.map(process_patient_biopsy, patient_biopsies_fixed, patient_biopsies_moving,
                              [path_biopsy_set] * len(patient_biopsies_fixed), [tile_path] * len(patient_biopsies_fixed))