from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from easydict import EasyDict
import numpy

#Implementation dataloader
class Biopsy_Dataset(Dataset):
    def __init__(self, config,transform=None):
        self.transform = transform
        self.config = config
        self.image_folder_HE = Path(config.paths_server.path_images_HE)
        self.image_folder_MUC = Path(config.paths_server.path_images_MUC)
        self.image_files_HE = [file for file in self.image_folder_HE.glob('*')]
        self.image_files_MUC = [file for file in self.image_folder_MUC.glob('*')]

    def __getitem__(self, idx):
        image_path_HE = self.image_files_HE[idx]
        image_path_MUC = self.image_files_MUC[idx]
        image_HE = Image.open(image_path_HE).convert('RGB')
        image_MUC = Image.open(image_path_MUC).convert('RGB')
        #labels = [float(value) for value in self.data.iloc[idx, 1:].values]

        # Apply transformations if specified
        if self.transform:
            image_HE = self.transform(image_HE)
            image_MUC = self.transform(image_MUC)
            image_HE = numpy.array(image_HE.permute(1, 2, 0))
            image_MUC = numpy.array(image_MUC.permute(1, 2, 0))

        return {'image_HE':image_HE, 'image_MUC' : image_MUC, 'image_path_HE': image_path_HE, 'image_path_MUC': image_path_MUC}

    def __len__(self):
        return len(self.image_files_MUC)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of the dataset
#dataset = Biopsy_Dataset(csv_file=location_data_csv, transform=transform)

# Create a data loader
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/config_dataloader.json"
    with open(config_filename,'r') as config_json:
        a = json.load(config_json)
        config = EasyDict(a)

    biopsy_data = Biopsy_Dataset(config,transform)

    plt.imshow(biopsy_data[0]["image_HE"])
    plt.show()
    plt.imshow(biopsy_data[0]["image_MUC"])
    plt.show()
