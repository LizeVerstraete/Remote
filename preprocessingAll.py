# Transform wsi images into small tiles in png format
import functions
import os
import json
from easydict import EasyDict

config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/config_preprocessing.json"
with open(config_filename, 'r') as config_json:
    a = json.load(config_json)
    config = EasyDict(a)

path_biopsies = config.path_biopsies
tile_path = config.tile_path
size = config.size

biopsy_sets = os.listdir(path_biopsies)
for biopsy_set in biopsy_sets:
    path_biopsy_set = os.path.join(path_biopsies,biopsy_set)
    tile_path = os.path.join(tile_path, biopsy_set)
    patient_biopsies = os.listdir(path_biopsy_set)
    patient_biopsies_fixed = sorted([biopsy for biopsy in patient_biopsies if 'HE' in biopsy and biopsy.lower().endswith('.mrxs')])
    patient_biopsies_moving = sorted([biopsy for biopsy in patient_biopsies if 'MUC' in biopsy and biopsy.lower().endswith('.mrxs')])

    #Make sure we have biopsy images from the patient for both HE and MUC type
    patients_fixed = [p.replace('HE','') for p in patient_biopsies_fixed]
    patients_moving = [p.replace('CDX2p_MUC2y_MUC5g_CD8dab','') for p in patient_biopsies_moving]
    patient_biopsies_fixed = sorted([p for p in patient_biopsies_fixed if p.replace('HE', '') in patients_moving])
    patient_biopsies_moving = sorted([p for p in patient_biopsies_moving if
                                        p.replace('CDX2p_MUC2y_MUC5g_CD8dab', '') in patients_fixed])

    index = -1
    for patient_biopsy_fixed in patient_biopsies_fixed:
        index += 1
        patient_biopsy_moving = patient_biopsies_moving[index]
        path_fixed_image = os.path.join(path_biopsy_set,patient_biopsy_fixed)
        path_moving_image = os.path.join(path_biopsy_set,patient_biopsy_moving)
        #check whether all the needed data is present (both the map and the .mrxs are needed)
        if os.path.exists(path_fixed_image.replace('.mrxs', '')):
            tile_path_fixed = os.path.join(tile_path,patient_biopsy_fixed.replace('.mrxs', ''))
            tile_path_moving = os.path.join(tile_path,patient_biopsy_moving.replace('.mrxs',''))
            if not os.path.exists(tile_path_fixed):
                os.makedirs(tile_path_fixed)
            if not os.path.exists(tile_path_moving):
                os.makedirs(tile_path_moving)
            dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(path_moving_image, path_fixed_image,
                                                                                  show_images=True)
            #Store tiles in a map corresponding to the specific patient and type of image
            functions.store_registered_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader,
                                                          moving_wsi_reader, size, tile_path_fixed, tile_path_moving)