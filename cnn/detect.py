import torch
from torchvision import transforms, models

from pathlib import Path
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from tkinter import filedialog
from tkinter import *

from PIL import Image
import cv2

from helpers.blocks_class import BlocksImage
from helpers import progress_bar


# Convolutional neural network (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

checkpoint = torch.load(Path('cnn.pt'), map_location='cpu')
model.load_state_dict(checkpoint)

model.eval()

data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def image_loader(loader, image):
    image = loader(image).float()
    # image = torch.tensor(image, requires_grad=True)
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


def mark_image(image):
    image_array = np.asarray(image)
    blocks = BlocksImage(image_array)
    # Initial call to print 0% progress
    progress_bar.printProgressBar(
        0, blocks.max_num_blocks(), prefix='Progress:',
        suffix='Complete', length=50)
    for num_block in range(blocks.max_num_blocks()):
        image_block = Image.fromarray(blocks.get_block(num_block))
        clase = np.argmax(model(
            image_loader(data_transforms, image_block)
        ).detach().numpy())
        if clase:
            coord = blocks.get_coord(num_block)
            cv2.rectangle(
                image_array, (coord[1], coord[0]),
                (coord[3], coord[2]), (0, 255, 0), 1)
        progress_bar.printProgressBar(
            num_block + 1, blocks.max_num_blocks(),
            prefix='Progress:', suffix='Complete', length=50)

    return Image.fromarray(image_array)


try:
    # Load cover image
    root = Tk()
    root.filename = filedialog.askopenfilename(
        initialdir="static/", title="Select file",
        filetypes=(
            ("jpg files", "*.jpg"), ("png files", "*.png"),
            ("all files", "*.*")))
    cover_image = Image.open(root.filename).convert('RGB')
    root.destroy()
    mark_image(cover_image).save('static/out.png')
except Exception as e:
    print("Error: ", e)
    print("The image file was not loaded")
