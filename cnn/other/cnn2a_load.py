from multiprocessing import freeze_support

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def classify_new_image():
    # Classify a new image using a pretrained model from the above training.

    # Location of the image we will classify.
    IMG_PATH = "/home/ernesto/this.png"

    # img = Image.open(IMG_PATH)
    # img.show()
    #
    # assert False

    # Pre-processing the new image using transform.
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )


    # Picture dataset.
    classify_dataset = datasets.ImageFolder(IMG_PATH, trans)
    # Create custom random sampler class to iter over dataloader.
    classify_loader = DataLoader(
        dataset=classify_dataset, batch_size=64, shuffle=True, num_workers=2)

    # # Check if gpu support is available
    # cuda_avail = torch.cuda.is_available()

    model = torch.load('cnn.pt')['state_dict']

# # if cuda is available, move the model to the GPU
# if cuda_avail:
#     model.cuda()

if __name__ == "__main__":
    classify_new_image()
