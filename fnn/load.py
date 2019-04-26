import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# Model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 8 * 8, 5000) 
        self.fc2 = nn.Linear(5000, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_checkpoint(filepath):

    model = Net()
    checkpoint = torch.load(Path(filepath), map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def image_loader(image_name):

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    image = Image.open(image_name)
    image = data_transforms(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


# Predict
file_pt_path = 'fnn600.pt'
path = '/home/ernesto/19_60.477.png'
model = load_checkpoint(file_pt_path)
print(np.argmax(model(image_loader(path)).detach().numpy()))
