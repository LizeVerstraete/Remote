# choose images with the typical colors to train the Kmeans on such that you have a proper color distribution to later use on different tiles with KNN
import numpy as np
from PIL import Image
import functions

green = Image.open(r'C:\\Users\\Lize\\Desktop\\School\\Master\\thesis\\tiles_by_hand\\green.png')
green = np.array(green.convert("RGB"))
yellow_purple = Image.open(r'C:\\Users\\Lize\\Desktop\\School\\Master\\thesis\\tiles_by_hand\\yellow_purple.png')
yellow_purple = np.array(yellow_purple.convert("RGB"))
brown_blue = Image.open(r'C:\\Users\\Lize\\Desktop\\School\\Master\\thesis\\tiles_by_hand\\brown_blue.png')
brown_blue = np.array(brown_blue.convert("RGB"))

width_max = max(green.shape[1], yellow_purple.shape[1], brown_blue.shape[1])
green = np.pad(green, ((0, 0), (0, width_max - green.shape[1]), (0, 0)), mode='constant')
yellow_purple = np.pad(yellow_purple, ((0, 0), (0, width_max - yellow_purple.shape[1]), (0, 0)), mode='constant')
brown_blue = np.pad(brown_blue, ((0, 0), (0, width_max - brown_blue.shape[1]), (0, 0)), mode='constant')

kmeans_image = np.concatenate((green, yellow_purple, brown_blue), axis=0)
data, labels, cluster_centers = functions.KMeans_image(kmeans_image)

# Choose a tile to compute the KNN on

all_colors = Image.open(r'C:\\Users\\Lize\\Desktop\\School\\Master\\thesis\\tiles_by_hand\\all_colors.png')
all_colors = np.array(all_colors.convert("RGB"))
functions.KNN(all_colors, data, labels, cluster_centers)
