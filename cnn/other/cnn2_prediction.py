from PIL import Image
import io
import requests

import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#
#
# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             # sum up batch loss
#             test_loss += F.nll_loss(output, target, reduction='sum').item()
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print(
#         '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))
#
#
# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument(
#         '--test-batch-size', type=int, default=1000, metavar='N',
#         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 0.01)')
#     parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                         help='SGD momentum (default: 0.5)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument(
#         '--log-interval', type=int, default=10, metavar='N',
#         help='how many batches to wait before logging training status')
#
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#
#     # Transforms
#     simple_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]
#     )
#
#     # Dataset
#     train_dataset = datasets.ImageFolder('static/train/', simple_transform)
#     valid_dataset = datasets.ImageFolder('static/valid/', simple_transform)
#
#     # Data loader
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset, batch_size=args.test_batch_size, shuffle=True,
#         num_workers=2)
#
#     test_loader = torch.utils.data.DataLoader(
#         dataset=valid_dataset, batch_size=args.test_batch_size, shuffle=False,
#         num_workers=2)
#
#     # Convolutional neural network (ResNet18)
#     model = models.resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 2)
#     model = model.to(device)
#     optimizer = optim.SGD(
#         model.parameters(), lr=args.lr, momentum=args.momentum)
#
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#
#     if (args.save_model):
#         torch.save(model.state_dict(), "cnn.pt")

if __name__ == '__main__':
    # main()
    # Model class must be defined somewhere

    img = Image.open('static/valid/text/text.4475.png')
    # response = requests.get(IMG_URL)
    # img = Image.open(io.BytesIO(response.content))
    # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    min_img_size = 8
    transform_pipeline = transforms.Compose([
        transforms.Resize(min_img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    img = Variable(img)
    model = torch.load('cnn.pt', map_location='cpu')
    prediction = model(img)
    prediction = prediction.data.numpy().argmax()
    print(prediction)
