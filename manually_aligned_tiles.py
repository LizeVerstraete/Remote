#Run visualizeAlignedTiles for a certain index for ... in ...[index:index+1]
#Adapt the location to a nice one you saw
from functions import load_image,crop_image,AffineWSITransformer,normalize_image
import matplotlib as plt
import os

location = (x, y)  # at base level 0

slide = load_image(path_fixed_image)
level = 5
_, [ymin, ymax, xmin, xmax] = crop_image(slide, level)  # at level 5 since lower memory-load
xmin = xmin * ((level + 1) ** 2)
xmax = xmax * ((level + 1) ** 2)
ymin = ymin * ((level + 1) ** 2)
ymax = ymax * ((level + 1) ** 2)

# xmax, ymax = slide.level_dimensions[0]

# DFBR transform is computed for level 7
# Hence it should be mapped to level 0 for AffineWSITransformer
dfbr_transform_level = 7
transform_level0 = dfbr_transform * [
    [1, 1, 2 ** dfbr_transform_level],
    [1, 1, 2 ** dfbr_transform_level],
    [1, 1, 1],
]

# Extract transformed region from the moving whole slide image
tfm = AffineWSITransformer(moving_wsi_reader, transform_level0)

# Extract region from the fixed whole slide image
fixed_tile = fixed_wsi_reader.read_rect(location, size, resolution=20,
                                        units="power")  # resolution at 20xzoom
moving_tile = tfm.read_rect(location, size, resolution=20, units="power")  # resolution at 20xzoom

fixed_tile = normalize_image(fixed_tile)

plt.subplot(1, 2, 1)
plt.imshow(fixed_tile)
plt.subplot(1, 2, 2)
moving_tile = normalize_image(moving_tile)
plt.imshow(moving_tile)
plt.show()


#STORE TILES
tile_dir = r'/esat/biomeddata/kkontras/r0786880/biopsy_data_manually_aligned_412/Janssen'

tile_dir_moving = os.path.join(tile_dir,os.path.splitext(patient_biopsy_moving)[0])
tile_dir_fixed = os.path.join(tile_dir,os.path.splitext(patient_biopsy_fixed)[0])
if not os.path.exists(tile_dir_fixed):
    os.mkdir(tile_dir_fixed)
if not os.path.exists(tile_dir_moving):
    os.mkdir(tile_dir_moving)
for x in range():
    for y in range():
        plt.subplot(1, 2, 1)
        plt.imshow(fixed_tile)
        plt.subplot(1, 2, 2)
        plt.imshow(moving_tile)
        plt.show()
        store = input('store this tile? [y]/[n]')
        if store == 'y':
            tile_name_fixed = os.path.join(tile_dir_fixed, '%d_%d' % (x, y))
            tile_name_moving = os.path.join(tile_dir_moving, '%d_%d' % (x, y))
            plt.imsave(tile_name_fixed + ".png", fixed_tile[])
            plt.imsave(tile_name_moving + ".png", moving_tile[])


