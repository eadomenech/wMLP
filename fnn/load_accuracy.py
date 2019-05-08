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
list_path = glob.glob('data/valid/17_90/*.png')


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


list_insert = []
for image_path in list_path:
    list_insert.append(predict(image_path))

for image_path in list_path:
    Image.open(image_path).save(
        image_path[:-4]+'.jpg', quality=20, optimice=True)

# Load images
list_path = glob.glob('data/valid/17_90/*.jpg')

list_extract = []
for image_path in list_path:
    list_extract.append(predict(image_path))

true_values = 0

for i in range(len(list_path)):
    # print("After: {}, Before: {}".format(list_insert[i], list_extract[i]))
    if list_insert[i] == list_extract[i]:
        true_values += 1

print("Accuracy: {}".format(true_values*100.0/len(list_path)))
