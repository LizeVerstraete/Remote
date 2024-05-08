import os
import json
from concurrent.futures import ThreadPoolExecutor
import functions

if __name__ == '__main__':
    config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
    with open(config_filename, 'r') as config_json:
        config = json.load(config_json)

    remove_existing_tiles = False
    path_biopsies = config["other_path_biopsies"]
    tile_path_init = config["tile_path_filtered"]
    size = config["variables"]["size_image"] #
    size = (int(size[0]/0.242534722222222), int(size[1]/0.242647058823529)) #convert micrometers to pixels (in 20x zoom)

    biopsy_sets = os.listdir(path_biopsies)

    def process_patient_biopsy(patient_biopsy_fixed, patient_biopsy_moving, path_biopsy_set, tile_path):
        print('ready to process ', patient_biopsy_fixed)
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
        functions.store_tiles(path_fixed_image,path_moving_image,size,tile_path_fixed,tile_path_moving)

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
