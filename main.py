import os
os.add_dll_directory(r'C:\Users\Lize\Downloads\openslide-win64-20230414\openslide-win64-20230414\bin')
import functions

# what do you want as an output
show_raw_image = True
show_cropped_image = True

# what is the path of your images
# Use HE as fixed and MUC as moving
mrxs_path_HE = r'C:\Users\Lize\Desktop\School\Master\thesis\biopsy data hospital\B-1912374_B01-01_HE.mrxs'
mrxs_path_MUC = r'C:\Users\Lize\Desktop\School\Master\thesis\biopsy data hospital\B-1912374_B01-01_CDX2p_MUC2y_MUC5g_CD8dab.mrxs'

# LOAD IMAGE
slide_MUC = functions.load_image(mrxs_path_MUC,show_raw_image)
slide_HE = functions.load_image(mrxs_path_HE,show_raw_image)

# CROP IMAGE
cropped_MUC, location_MUC_cropped = functions.crop_image(slide_MUC, level=5, show_cropping=show_cropped_image)
cropped_HE, location_HE_cropped = functions.crop_image(slide_HE, level=5, show_cropping=show_cropped_image)
#
# # SPLIT IN TILES
# tiles_MUC = functions.split_in_tiles(cropped_MUC,slide_MUC,100,0, location_MUC_cropped)
# tiles_HE = functions.split_in_tiles(cropped_HE,slide_HE,100,0, location_HE_cropped)
#
# # LABEL TILES
# labeled_tiles = functions.label_tiles(tiles_MUC)
#
# if __name__ == '__main__':
#     aligned_MUC, aligned_HE = functions.allign(mrxs_path_MUC, mrxs_path_HE)

(print('end'))

# CALCULATION OF REQUIRED RESOLUTION TO GET ZOOM-LEVEL x20
# Resolution = 1/MMP
# Level0: 0.2425 MMP
# Objective_power = 20
# MMP at 20 zoom =  0.2425/(2**20)*(10**6) = 0.2313
#Resolution at 20 zoom = 1/0.2313 = 4.3234
