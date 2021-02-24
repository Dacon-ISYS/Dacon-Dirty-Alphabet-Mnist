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

from torchvision import transforms

import random
from efficientnet_pytorch import EfficientNet

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_path = 'dataset2/'

## Hyper parameter
num_epochs = 60
batch_size = 8


# Dirty MNIST Argumentation 자동으로 해주는 것
class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike, # 경로
        image_ids: os.PathLike, # image 경로
        transforms: Sequence[Callable] # 함수 call
    ) -> None:
        # 생성자 변수들
        self.dir = dir # 경로
        self.transforms = transforms
        self.labels = {} # label -> dictionary

        # csv 라이브러리 이용해서 data 읽어오기
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    # image 갯수 리턴
    def __len__(self) -> int:
        return len(self.image_ids)

    # image 읽어오기
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
    submit = pd.read_csv(data_path+'sample_submission.csv')

    # augmentation 함수
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(contrast=(0.8, 1.4)), #, brightness=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2), shear=15),
#        transforms.RandomAffine(random.randint(-15, 15)),
        transforms.RandomRotation(15),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(contrast=(1, 1.8)),  # , brightness=(0.8, 1.2)),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    trainset = MnistDataset(data_path+'train/', data_path+'label/dirty_mnist_2nd_answer.csv', transforms_train)
    testset = MnistDataset(data_path+'test/', data_path+'sample_submission.csv', transforms_test)

    train_loader = DataLoader(trainset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)

    # device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=26)
    model = nn.DataParallel(model)
    print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

    # set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criterion = nn.MultiLabelSoftMarginLoss()
    valid_result = []  # validation 결과 모아서 보기

    # set hyper parameter options
    for epoch in range(1, num_epochs):
        # model training - 매 epoch 마다
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device); targets = targets.to(device) # data, label DEVICE 로 DATA 보내기

            optimizer.zero_grad() # 최적화 함수에 대한 미분 진행 여부 설정
            outputs = model(images) # model 통과 결과
            loss = criterion(outputs, targets) # target값과의 loss 비교를 통한 crossentropy
            loss.backward() # 역방향 전파
            optimizer.step() # 미분 반복 진행
            lr_scheduler.step() # scheduler step

            # 10번째 학습마다 accuracy
            if (i + 1) % 10 == 0:
                outputs = outputs > 0.5
                acc = (outputs == targets).float().mean()
                print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')
        ############### epoch 한번 training 완료 ######################

        # epoch 15번째부터 테스트
        if epoch >=30:
            print('test mode start epoch ', epoch)

            # test 위한 Evaluation
            model.eval() # 평가 모드로 모델 변경
            batch_size = test_loader.batch_size
            batch_index = 0

            # test 결과
            for i, (images, targets) in enumerate(test_loader):
                images = images.to(device); targets = targets.to(device) # data 보내기
                outputs = model(images) # model 통과 결과
                outputs = outputs > 0.5
                batch_index = i * batch_size

                # submit 파일에 정답 쓰기
                submit.iloc[batch_index:batch_index + batch_size, 1:] = \
                    outputs.long().squeeze(0).detach().cpu().numpy()
            # submit 파일 저장
            submit.to_csv('result_submit3/efficient_submit'+str(epoch)+'.csv', index=False)
            # torch.save(model.state_dict(), 'result_submit3/v4_efficientb7_epoch' + str(epoch) + ".pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'result_submit3/efficientb7_epoch_0223_linux_' + str(epoch) + ".pt")
