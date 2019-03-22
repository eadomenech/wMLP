import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn as nn
from torch.autograd import Variable

# Convolutional neural network (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

checkpoint = torch.load(Path('cnn.pt'), map_location='cpu')
model.load_state_dict(checkpoint)

model.eval()

# resnet_model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
# resnet_model.eval()

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# # An instance of your model.
# img_pil = Image.open("/home/ernesto/this.png")

image = Image.open(Path('/home/ernesto/this.png'))
input = trans(image)

# # img_pil.show()
# img_tensor = preprocess(img_pil).float()
# img_tensor = img_tensor.unsqueeze_(0)

fc_out = model(Variable(input))

output = fc_out.detach().numpy()
print(output.argmax())
