import os
from typing import Tuple, Sequence, Callable
import csv
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchvision import transforms
import random
from efficientnet_pytorch import EfficientNet

seed = 77
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Dirty MNIST Argumentation 자동으로 해주는 것
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


transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(random.randint(0, 360)),
    transforms.ColorJitter(contrast=(0.2, 3)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_validation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

trainset = MnistDataset('data/train', 'data/dirty_mnist_answer.csv', transforms_train)
validationset = MnistDataset('data/validation', 'data/dirty_mnist_answer-validation.csv', transforms_test)
testset = MnistDataset('data/test', 'data/sample_submission.csv', transforms_test)

train_loader = DataLoader(trainset, batch_size=8, num_workers=2)
validation_loader = DataLoader(validationset, batch_size=8, num_workers=2)
test_loader = DataLoader(testset, batch_size=8, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=26)
model = nn.DataParallel(model).to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

num_epochs = 60

model.train()
test_result = []  # validation 결과 모아서 보기

path = "./efficientnets/"

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')

    model.eval()
    batch_size = test_loader.batch_size
    batch_index = 0
    valid_count = 0

    for i, (images, targets) in enumerate(validation_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        result = str(epoch) + " : " + str(loss.item())
        test_result.append(result)
        outputs = outputs > 0.5
        batch_index = i * batch_size
        acc = (outputs == targets).float().mean()

        valid_count += 1
        if valid_count == 5:
            break

        print("valid", f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')

    torch.save(model, path + "epoch" + str(epoch) + ".pt")
    model.train()

print(test_result)