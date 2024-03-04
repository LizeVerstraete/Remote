from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import dataloader
import json

config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
with open(config_filename, 'r') as config_json:
    config = json.load(config_json)

#Start of the training of the model, might want to put this in separate file
biopsy_dataset = dataloader.Biopsy_Dataset(config,transform)
data = DataLoader(biopsy_dataset,config.variables.batch_size)

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