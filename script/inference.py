import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import argparse
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from PIL import Image

input_path = '../data/'
num_epochs = 30
batch_size = 32

with open(input_path + 'classes.txt') as f:
    classes = [x.strip() for x in f.readlines()]

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((244,244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
}

class birdDataset(Dataset):
    def __init__(self, datatype, datalist):
        self.transform = data_transforms[datatype]
        self.datatype = datatype
        try:
            if datatype is 'train':
                with open(input_path + datalist) as f:
                    train_data = [x.split() for x in f.readlines()]
                    self.images = [f'../data/{datatype}/{data[0]}' for data in train_data]
                    self.labels = [int(data[1].split('.')[0])-1 for data in train_data]
                
                assert len(self.images) == len(self.labels), 'mismatched length!'
            elif datatype is 'test':
                with open(input_path + datalist) as f:
                    self.images = [x.strip() for x in f.readlines()]  # all the testing images
        except:
            print("Unexpected error:", sys.exc_info()[0])
    
    def __getitem__(self, index):
        imgpath = self.images[index]        
        image = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.datatype is 'train':
            label = self.labels[index]
            return image, label
        elif self.datatype is 'test':
            return image

    def __len__(self):
        return len(self.images)

image_datasets = {
    'train': birdDataset(datatype='train', datalist='training_labels.txt'),
    'test': birdDataset(datatype='test', datalist='testing_img_order.txt'),
}

dataloaders = {
    'train':
    DataLoader(dataset=image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'test':
    DataLoader(dataset=image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0),
}

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    plt.savefig('img.png')

def train_model(model, dataloader, criterion, optimizer, dataset_len, num_epochs=10, device='cpu'):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # for phase in ['train', 'validation']:
        
        # if phase == 'tra in':
        model.train()
        # else:
        #     model.eval()

        running_loss = 0.0
        running_corrects = 0

        # for inputs, labels in dataloaders[phase]:
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / len(image_datasets[phase])
        # epoch_acc = running_corrects.double() / len(image_datasets[phase])
        epoch_loss = running_loss / dataset_len
        epoch_acc = running_corrects.double() / dataset_len

        print('loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print()
    return model

def predict(model, device):
    with open('../data/testing_img_order.txt') as f:
        test_images_paths = [x.strip() for x in f.readlines()]  # all the testing images
    
    submission = []
    for image_f in test_images_paths:
        image = Image.open(input_path + 'test/' + image_f)
        batch_image = torch.stack([data_transforms['test'](image).to(device)])
        predict = model(batch_image)
        pred_prob = torch.argmax(predict).tolist()
        submission.append([image_f, classes[pred_prob]])

    np.savetxt('answer.txt', submission, fmt='%s')


    # test_batch = torch.stack([data_transforms['test'](image).to(device)
    #                           for image in image_list])

    # predict = model(test_batch)
    # pre_prob = torch.argmax(predict).tolist()
    
    # print(pre_prob)
    return

    submission = []
    for image_f in test_images:  # image order is important to your result
        image = Image.open('../data/test/' + image_f).convert('RGB')
        if transform is not None:
            image = transform(image)
        print(image.shape)
        predicted_class = model(image)  # the predicted category
        print("Predict1:", predicted_class)
        submission.append([image_f, predicted_class])

    np.savetxt('answer.txt', submission, fmt='%s')

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='save state of model')
    parser.add_argument('--load', type=str, default=None, help='load model with state file')
    return parser

def main(args, device):
    # train_set = birdDataset(root='../data/', transform=data_transform['train'])
    # train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=4)

    for images, labels in dataloaders['train']:
        # print("Image:", images)
        # print('Label:', labels)
        imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % labels[j] for j in range(len(labels))))
        break

    # device = torch.device('cpu')

    model = models.resnet50(pretrained=True).to(device)

    # for param in model.parameters():
        # param.requires_grad = False

    # model.fc = nn.Sequential(
    #         nn.Linear(2048, 1024),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(1024, len(classes))).to(device)

    if args.load:
        model.load_state_dict(torch.load('./best.pt'))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        model = train_model(model, dataloaders['train'], criterion, optimizer, len(image_datasets['train']),
                            num_epochs=num_epochs, device=device)

        if args.save:
            torch.save(model.state_dict(), './best.pt')

    model.eval()
    print('Start predict...')
    predict(model, device=device)
    print("Finish!")
    # for img_f in train_images:
    #     img = Image.open('../data/train/' + img_f).convert('RGB')
        # print(img.size())
        # print(transform(img).size())

        # convert_tensor = transforms.ToTensor()
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                       shuffle=True, num_workers=2)

    # print(trainset)

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    # Select GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(args, device)
