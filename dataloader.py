from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from easydict import EasyDict
import numpy
import functions

#Implementation dataloader
class Biopsy_Dataset(Dataset):
    def __init__(self, config,transform=None):
        self.transform = transform
        self.config = config
        self.image_folder = Path(config.paths_server.path_images)
        self.image_files_HE = [file for file in self.image_folder.glob('*') if
         file.suffix.lower() == '.mrxs' and "HE" in file.stem]
        self.image_files_MUC = [file for file in self.image_folder.glob('*') if
         file.suffix.lower() == '.mrxs' and "MUC" in file.stem]
        self.path_kmeans_image = config.paths_server.path_kmeans_image

    def __getitem__(self, idx):
        image_path_fixed = self.image_files_HE[idx]
        image_path_moving = self.image_files_MUC[idx]
        #image = Image.open(image_path_fixed).convert('RGB')
        #labels = [float(value) for value in self.data.iloc[idx, 1:].values]

        # Apply transformations if specified
        #if self.transform:
        #    image = self.transform(image)
        #    image = numpy.array(image.permute(1, 2, 0))

        #return {'image': image, 'image_path_fixed': image_path_fixed,'image_path_moving': image_path_moving}
        return {'image_path_fixed': image_path_fixed,'image_path_moving': image_path_moving}

    def __len__(self):
        return len(self.image_files)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of the dataset
#dataset = Biopsy_Dataset(csv_file=location_data_csv, transform=transform)

# Create a data loader
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    config_filename = r"\esat\biomeddata\kkontras\r0786880\models\remote\Configuration.json"
    with open(config_filename,'r') as config_json:
        a = json.load(config_json)
        config = EasyDict(a)

    biopsy_data = Biopsy_Dataset(config,transform)

    # open the default image to be used to calculate the Kmeans
    kmeans_image = Image.open(biopsy_data.path_kmeans_image)
    data, labels, cluster_centers = functions.KMeans_image(kmeans_image)

    for batch in biopsy_data:
        #plt.imshow(batch['image'])
        #plt.show()
        # Functions to setup a dataset stored on the server

        #Align images
        dfbr_transform, fixed_wsi_reader, moving_wsi_reader = functions.align(batch["image_path_moving"], batch["image_path_fixed"], show_images=False)
        tile_names = functions.store_registered_tiles(batch["image_path_fixed"], dfbr_transform, fixed_wsi_reader, moving_wsi_reader, biopsy_data.size, biopsy_data.paths_server.tile_dir_fixed, biopsy_data.paths_server.tile_dir_moving)
        #soft_labels = functions.soft_labeling(tile_names, data, labels, cluster_centers, tile_dir_moving)
        #functions.create_dataset(tile_names, soft_labels, location_data_csv)  # generates csv file containing paths to all tiles and there softlabels
