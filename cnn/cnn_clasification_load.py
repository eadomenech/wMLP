import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Convolutional neural network (ResNet18)
model = models.densenet201(pretrained=True)
print(model)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 11)

checkpoint = torch.load(Path('cnn_clasification_desenet201.pt'), map_location='cpu')
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

print(np.argmax(model(image_loader(data_transforms, '/media/ernesto/Ernesto/cnn/data/valid/19_60/19_60.6.png')).detach().numpy()))
print(model(image_loader(data_transforms, '/media/ernesto/Ernesto/cnn/data/valid/19_60/19_60.6.png')))