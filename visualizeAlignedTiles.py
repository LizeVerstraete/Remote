# Transform wsi images into small tiles in png format
import functions
import os
import json

config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
with open(config_filename, 'r') as config_json:
    config = json.load(config_json)

remove_existing_tiles = False
path_biopsies = config["path_biopsies"]
tile_path_init = config["tile_path_filtered_aligned"]
size = config["variables"]["size_image"] #
size = (int(size[0]/0.242534722222222), int(size[1]/0.242647058823529)) #convert micrometers to pixels (in 20x zoom)
size = [8000,8000]

#WILL ONLY PROCESS JANSSEN, REWRITE AGAIN!!!

biopsy_sets = os.listdir(path_biopsies)
#for biopsy_set in biopsy_sets:
biopsy_set = 'Janssen'
path_biopsy_set = path_biopsies
tile_path = os.path.join(tile_path_init, biopsy_set)
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
    if os.path.exists(path_fixed_image.replace('.mrxs', '')) and os.path.exists(path_moving_image.replace('.mrxs', '')):
        tile_path_fixed = os.path.join(tile_path,patient_biopsy_fixed.replace('.mrxs', ''))
        tile_path_moving = os.path.join(tile_path,patient_biopsy_moving.replace('.mrxs',''))
        #Create maps to store the tiles per dataset and per patient -> can be adapted to store per label (MSS vs MSI)

        # #Clear maps if they allready contain tiles from previous calculations
        if remove_existing_tiles:
            if os.listdir(tile_path_fixed):
                for file in os.listdir(tile_path_fixed):
                    file_path = os.path.join(tile_path_fixed,file)
                    os.remove(file_path)
            if os.listdir(tile_path_moving):
                for file in os.listdir(tile_path_moving):
                    file_path = os.path.join(tile_path_moving,file)
                    os.remove(file_path)

        dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(path_moving_image, path_fixed_image, show_images=True)


    #functions.bspline(dfbr_transform, fixed_wsi_reader, moving_wsi_reader, [24500,146512], size)

        functions.show_normalized_aligned_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader, moving_wsi_reader,
                                        size, tile_path_fixed, tile_path_moving)