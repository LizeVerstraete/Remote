# Transform wsi images into small tiles in png format
import functions
import json
import os
config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
with open(config_filename, 'r') as config_json:
    config = json.load(config_json)

path_fixed_image = "/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Microbiome CRC/B-1860002_01-04_HE_20221018.mrxs"
path_moving_image = "/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Microbiome CRC/B-1860002_01-04_CDX2p_MUC2y_MUC5g_CD8dab_20221018.mrxs"
tile_path_moving = "/esat/biomeddata/kkontras/r0786880/tmp_data/Microbiome CRC/BB-1860002_01-04_HE_20221018"
tile_path_fixed = "/esat/biomeddata/kkontras/r0786880/tmp_data/Microbiome CRC/B-1860002_01-04_CDX2p_MUC2y_MUC5g_CD8dab_20221018"
if not os.path.exists(tile_path_fixed):
    os.makedirs(tile_path_fixed)
if not os.path.exists(tile_path_moving):
    os.makedirs(tile_path_moving)

size = config["variables"]["size_image"] #
size = (int(size[0]/0.242534722222222), int(size[1]/0.242647058823529)) #convert micrometers to pixels (in 20x zoom)

functions.store_normalized_tiles(path_fixed_image, path_moving_image, size, tile_path_fixed, tile_path_moving)