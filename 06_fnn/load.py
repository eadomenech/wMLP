import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import random
from io import BytesIO


'''
16_120 -> 2
17_90 -> 1
19_53 -> 4
19_60 -> 4
19_130 -> 2
20_130 -> 5
28_94 -> 7
28_120 -> 2
34_130 -> 8
'''


# Load images
list_path = glob.glob('data/valid/34_130/*.png')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 8 * 8, 5000)
        self.fc2 = nn.Linear(5000, 9)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
checkpoint = torch.load(Path('fnn600_with_jpeg_compression_transform.pt'), map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

def randomJpegCompression(image):
    qf = random.randrange(10, 30)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

data_transforms = transforms.Compose(
    [
        transforms.Lambda(randomJpegCompression),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def predict(image_path):
    image = Image.open(image_path)
    image_tensor = data_transforms(image)

    # PyTorch pretrained models expect the Tensor dims to be
    # (num input imgs, num color channels, height, width).
    # Currently however, we have (num color channels, height, width);
    # let's fix this by inserting a new axis.
    image_tensor = image_tensor.unsqueeze(0)

    output = model(Variable(image_tensor))

    return np.argmax(output.detach().numpy())


for image_path in list_path:
    print(predict(image_path))