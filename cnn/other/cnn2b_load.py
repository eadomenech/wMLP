import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn as nn

# Convolutional neural network (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

checkpoint = torch.load(Path('cnn.pt'), map_location='cpu')
model.load_state_dict(checkpoint)

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

image = Image.open(Path('/home/ernesto/this.png'))

input = trans(image)

input = input.view(64, 3, 8, 8)

output = model(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

if (prediction == 0):
    print('not_test')
else:
    print('test')
