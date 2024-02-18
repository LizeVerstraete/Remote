from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import functions

#Variables
#path to original images -> for now just two but can be extended to all of them by reading out the existing maps in Janssen
path_fixed_image = r'\esat\smcdata\tempusers\r0786880\no_backup\Biopsies\Janssen\B-2028243_B1-12_HE.mrxs'
path_moving_image = r'\esat\smcdata\tempusers\r0786880\no_backup\Biopsies\Janssen\B-2028243_B1-12_CDX2p_MUC2y_MUC5g_CD8dab.mrxs'
#path where adjusted intermediate image-data will be stored such as the tiles
tile_dir_fixed = r'\esat\biomeddata\kkontras\r0786880\data\tiles_fixed'
tile_dir_moving = r'\esat\biomeddata\kkontras\r0786880\data\tiles_moving'
location_data_csv = r"\esat\biomeddata\kkontras\r0786880\data\dataset.csv"
size = [512,512]

#open the default image to be used to calculate the Kmeans
kmeans_image = Image.open(r'\esat\biomeddata\kkontras\r0786880\data\kmeans.png')
data, labels, cluster_centers = functions.KMeans_image(kmeans_image)

#Functions to setup a dataset stored on the server
dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(path_moving_image, path_fixed_image, show_images=False)
tile_names = functions.store_registered_tiles(path_fixed_image, dfbr_transform, fixed_wsi_reader, moving_wsi_reader, size, tile_dir_fixed, tile_dir_moving)
soft_labels = functions.soft_labeling(tile_names, data, labels, cluster_centers,tile_dir_moving)
functions.create_dataset(tile_names, soft_labels, location_data_csv) #generates csv file containing paths to all tiles and there softlabels

#Implementation dataloader
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        labels = [float(value) for value in self.data.iloc[idx, 1:].values]

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': labels}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of the dataset
dataset = CustomDataset(csv_file=location_data_csv, transform=transform)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)