import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Convolutional neural network (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

checkpoint = torch.load(Path('cnn.pt'), map_location='cpu')
model.load_state_dict(checkpoint)

model.eval()


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

print(np.argmax(model(image_loader(data_transforms, '/home/ernesto/not_text10007.png')).detach().numpy()))
