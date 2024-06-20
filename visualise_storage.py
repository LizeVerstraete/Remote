import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from functions import load_image, crop_image
import re
import os
import numpy as np
patient_id = "B-2027329_B1-4"
store = True
def extract_coordinates(filename):
    # Define a regex pattern to extract the two numbers at the end of the filename
    pattern = r'(\d+)_(\d+)\.png$'
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    if match:
        # Extract the coordinates
        coord1, coord2 = match.groups()
        return int(coord1), int(coord2)
    else:
        raise ValueError("Filename does not match the expected pattern")

level = 5
path_MUC = os.path.join(r'/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Janssen',patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
path_HE = os.path.join(r'/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Janssen',patient_id + '_HE.mrxs')
path_tiles_MUC = os.path.join(r'/esat/biomeddata/kkontras/r0786880/biopsy_data_bigger_dataset_412_entropy_norm11/Janssen',patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab")
path_tiles_HE = os.path.join(r'/esat/biomeddata/kkontras/r0786880/biopsy_data_bigger_dataset_412_entropy_norm11/Janssen',patient_id + "_HE")
tilesMUC = os.listdir(path_tiles_MUC)
tilesHE = os.listdir(path_tiles_HE)
slide_MUC = load_image(path_MUC)
slide_HE = load_image(path_HE)
_,[yminMUC,ymaxMUC,xminMUC,xmaxMUC] = crop_image(slide_MUC,5,True)
_,[yminHE,ymaxHE,xminHE,xmaxHE] = crop_image(slide_HE,5,True)
cropped_MUC = np.array(slide_MUC.read_region((xminMUC * 32,yminMUC * 32),5,(xmaxMUC-xminMUC,ymaxMUC-yminMUC)))
cropped_HE = np.array(slide_HE.read_region((xminHE * 32,yminHE * 32),5,(xmaxHE-xminHE,ymaxHE-yminHE)))
if store:
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/cropped_MUC.pdf",zoom(cropped_MUC, (0.25,0.25,1)))
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/cropped_HE.pdf",zoom(cropped_HE, (0.25,0.25,1)))
xminHE = xminHE * 32
xmaxHE = xmaxHE * 32
yminHE = yminHE * 32
ymaxHE = ymaxHE * 32
xminMUC = xminMUC * 32
xmaxMUC = xmaxMUC * 32
yminMUC = yminMUC * 32
ymaxMUC = ymaxMUC * 32

for tile in tilesMUC:
    x,y = extract_coordinates(tile)
    x = (x-xminMUC)/32
    y = (y-yminMUC)/32
    cropped_MUC[int(y):int(y+412/32),int(x):int(x+412/32)] = [0, 0, 139, 255]
for tile in tilesHE:
    x,y = extract_coordinates(tile)
    x = (x-xminHE)/32
    y = (y-yminHE)/32
    cropped_HE[int(y):int(y+412/32),int(x):int(x+412/32)] = [0, 0, 139, 255]
plt.figure()
plt.imshow(cropped_MUC)
plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
plt.show()
plt.figure()
plt.imshow(cropped_HE)
plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
plt.show()
if store:
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/savingsMUC.png", zoom(cropped_MUC, (0.25,0.25,1)))

    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/savingsHE.pdf",zoom(cropped_HE, (0.25,0.25,1)))
# ADD MANUAL ADDITIONAL FILL
# ADD visuals patients

train_patients = ['B-1986943_B8_HE', 'B-1989502_B6_HE', 'B-1986096_B4_HE', 'B-2027329_B1-11_HE', 'B-1986183_B7_HE', 'B-1986096_B5_HE', 'B-1989502_B5_HE', 'B-1989502_B8_HE', 'B-1986096_B6_HE', 'B-1986183_B6_HE', 'B-1993721_B9_HE', 'B-1993721_B5_HE', 'B-2027329_B1-10_HE', 'B-1993721_B6_HE', 'B-2027329_B1-5_HE', 'B-1986943_B6_HE', 'B-1989502_B11_HE', 'B-1989502_B7_HE', 'B-1993721_B7_HE', 'B-1986183_B8_HE']
test_patients = ['B-1989502_B9_HE', 'B-1989502_B3_HE', 'B-1986096_B3_HE']
val_patients = ['B-1986943_B7_HE', 'B-2027329_B1-4_HE']

for patient_id in train_patients:
    patient_id = patient_id.replace('_HE', '')
    path_MUC = os.path.join(r'/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Janssen',
                            patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
    slide_MUC = load_image(path_MUC)
    _, [yminMUC, ymaxMUC, xminMUC, xmaxMUC] = crop_image(slide_MUC, 5, False)
    cropped_MUC = np.array(
        slide_MUC.read_region((xminMUC * 32, yminMUC * 32), 5, (xmaxMUC - xminMUC, ymaxMUC - yminMUC)))
    xminMUC = xminMUC * 32
    xmaxMUC = xmaxMUC * 32
    yminMUC = yminMUC * 32
    ymaxMUC = ymaxMUC * 32
    path_tiles_MUC = os.path.join(
        r'/esat/biomeddata/kkontras/r0786880/biopsy_data_bigger_dataset_412_entropy_norm11/Janssen',
        patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab")
    tilesMUC = os.listdir(path_tiles_MUC)
    for tile in tilesMUC:
        x, y = extract_coordinates(tile)
        x = (x - xminMUC) / 32
        y = (y - yminMUC) / 32
        cropped_MUC[int(y):int(y + 412 / 32), int(x):int(x + 412 / 32)] = [0, 0, 139, 255]
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/savingsMUC" + patient_id + ".pdf",
               zoom(cropped_MUC, (0.25, 0.25, 1)))


for patient_id in test_patients:
    patient_id = patient_id.replace('_HE','')
    path_MUC = os.path.join(r'/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Janssen',
                            patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
    slide_MUC = load_image(path_MUC)
    _,[yminMUC,ymaxMUC,xminMUC,xmaxMUC] = crop_image(slide_MUC,5,True)
    cropped_MUC = np.array(
        slide_MUC.read_region((xminMUC * 32, yminMUC * 32), 5, (xmaxMUC - xminMUC, ymaxMUC - yminMUC)))
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/testset"+patient_id+".pdf",zoom(cropped_MUC, (0.25,0.25,1)))

for patient_id in val_patients:
    patient_id = patient_id.replace('_HE','')
    path_MUC = os.path.join(r'/esat/smcdata/tempusers/r0786880/no_backup/Biopsies/Janssen',
                            patient_id + "_CDX2p_MUC2y_MUC5g_CD8dab.mrxs")
    slide_MUC = load_image(path_MUC)
    _,[yminMUC,ymaxMUC,xminMUC,xmaxMUC] = crop_image(slide_MUC,5,True)
    cropped_MUC = np.array(
        slide_MUC.read_region((xminMUC * 32, yminMUC * 32), 5, (xmaxMUC - xminMUC, ymaxMUC - yminMUC)))
    plt.figure()
    plt.imsave("/esat/biomeddata/kkontras/r0786880/results/temp/valset"+patient_id+".pdf",zoom(cropped_MUC, (0.25,0.25,1)))