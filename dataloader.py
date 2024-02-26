from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from easydict import EasyDict
import torch.nn as nn
import numpy
import torch
import torch.optim as optim

class Biopsy_Dataset(Dataset):
    #Example code for one patient
    # def __init__(self, config,transform=None):
    #     self.transform = transform
    #     self.config = config
    #     self.image_folder_HE = Path(config.paths_server.path_images_HE)
    #     self.image_folder_MUC = Path(config.paths_server.path_images_MUC)
    #     self.image_files_HE = [file for file in self.image_folder_HE.glob('*')]
    #     self.image_files_MUC = [file for file in self.image_folder_MUC.glob('*')]

    def __init__(self, config,transform=None):
        #super(Biopsy_Dataset).__init__()
        self.transform = transform
        self.config = config
        self.tile_folders = sorted([file for file in Path(config.tile_path).glob('*')])
        self.image_files_HE = []
        self.image_files_MUC = []
        for tile_folder in self.tile_folders:
            image_files_HE_current = sorted([file for file in tile_folder.glob('*HE*')])
            self.image_files_HE.extend(image_files_HE_current)
            image_files_MUC_current = sorted([file for file in tile_folder.glob('*MUC*')])
            self.image_files_MUC.extend(image_files_MUC_current)
        assert len(self.image_files_HE) == len(self.image_files_MUC), "You need equally much HE and MUC images"

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
            if check_permute(image_HE):
                image_HE = image_HE.permute(1, 2, 0)
            if check_permute(image_MUC):
                image_MUC = image_MUC.permute(1, 2, 0)

        return {'image_HE':image_HE, 'image_MUC' : image_MUC, 'image_path_HE': image_path_HE, 'image_path_MUC': image_path_MUC}

    def __len__(self):
        return len(self.image_files_MUC)


def check_permute(image):
    #If image is in (C,H,W) format, it needs to be rearranged to (H,W,C) format
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        return True
    else:
        return False

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
with open(config_filename, 'r') as config_json:
    a = json.load(config_json)
    config = EasyDict(a)

biopsy_dataset = Biopsy_Dataset(config, transform)
#Give an example plot of what a set of HE - MUC image looks like
if __name__ == "__main__":
    plt.imshow(biopsy_dataset[0]["image_HE"])
    plt.show()
    plt.imshow(biopsy_dataset[0]["image_MUC"])
    plt.show()

#Start of the training of the model, might want to put this in separate file
biopsy_dataset = Biopsy_Dataset(config,transform)
dataloader = DataLoader(biopsy_dataset,config.variables.batch_size)

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=config.variables.lr)

MSE = nn.MSELoss()

#CHECK WHETHER THIS WAY OF LOADING HE AND MUC IN THE FOR LOOP WORKS
for epoch in range(config.variables.epochs):
    for HE_images, MUC_images in dataloader:
        HE_images = HE_images.to(device)
        MUC_images = MUC_images.to(device)
        generated_MUC = model(HE_images)
        loss = MSE(generated_MUC,MUC_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()