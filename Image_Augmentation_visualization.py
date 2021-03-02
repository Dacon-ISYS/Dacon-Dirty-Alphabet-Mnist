import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import glob
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

data_path = 'D:\dataset\Dacon\dirty_mnist\\'

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

if __name__ == "__main__":
    # raw data 20개만 뽑아오기
    file_list = glob.glob(data_path+'train/*.png')[:20]
    print(len(file_list))

    # device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform data
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(contrast=(1, 2)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    # 학습데이터
    trainset = MnistDataset(data_path+'train/', data_path+'label/dirty_mnist_2nd_answer_90_or_270_train.csv', transforms_train)
    train_loader = DataLoader(trainset, batch_size=1)

    # image load 및 확인 - row data vs augmentation data
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device); targets = targets.to(device)  # data, label DEVICE 로 DATA 보내기
        img = np.squeeze(images.cpu().numpy().transpose(3,2,1,0), 3)
        print(img.shape)

        raw_img = cv2.imread(file_list[i])
        cv2.imshow('raw', raw_img)
        cv2.imshow('Augmentation',img)
        cv2.waitKey()
        if i== len(file_list)-1:
            break