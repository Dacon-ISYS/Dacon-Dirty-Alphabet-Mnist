import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torchvision.models import resnet50, vgg16, alexnet, vgg11, squeezenet1_0

class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


transforms_test = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


testset = MnistDataset('data/test', 'data/sample_submission.csv', transforms_test)

test_loader = DataLoader(testset, batch_size=128, num_workers=2)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = torch.load("efficientnets-b7/epoch51.pt")  # change this
model2 = torch.load("efficientnets-b6/epoch51.pt")  # change this
model3 = torch.load("efficientnets-b0/epoch49")  # change this
model4 = torch.load("efficientnets-b1/epoch50")  # change this

submit = pd.read_csv('data/sample_submission.csv')

model.eval()
model2.eval()
model3.eval()
model4.eval()
batch_size = test_loader.batch_size
batch_index = 0

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        outputs2 = model2(images)
        outputs3 = model3(images)
        outputs4 = model4(images)

        summation = outputs*0.6 + outputs2*0.2 + outputs3*0.1 + outputs4*0.1
        summation = summation > 0.5
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index + batch_size, 1:] = \
            summation.long().squeeze(0).detach().cpu().numpy()
        if i % 10 == 0:
            print(i)

submit.to_csv('submit.csv', index=False)
