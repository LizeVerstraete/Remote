# Transform wsi images into small tiles in png format
import functions

path_fixed_image = "/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Microbiome CRC/B-1860002_01-04_HE_20221018.mrxs"
path_moving_image = "/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Microbiome CRC/B-1860002_01-04_CDX2p_MUC2y_MUC5g_CD8dab_20221018.mrxs"
tile_path_moving = "/esat/biomeddata/kkontras/r0786880/data/Microbiome CRC/BB-1860002_01-04_HE_20221018"
tile_path_fixed = "/esat/biomeddata/kkontras/r0786880/data/Microbiome CRC/B-1860002_01-04_CDX2p_MUC2y_MUC5g_CD8dab_20221018"
size = [512,512]

dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(path_moving_image, path_fixed_image, show_images=True)
tile_names = functions.store_registered_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader,moving_wsi_reader,size, tile_path_fixed,tile_path_moving)
