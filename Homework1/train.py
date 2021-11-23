import os
import sys
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from PIL import Image

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

input_path = './data/'

with open(input_path + 'classes.txt') as f:
    classes = [x.strip() for x in f.readlines()]

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
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
            if datatype is 'train' or datatype is 'valid':
                with open(input_path + datalist) as f:
                    train_data = [x.split() for x in f.readlines()]
                    self.images = [f'./data/{datatype}/{data[0]}' for data in train_data]
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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    plt.savefig('img.png')

def show_acc(history, phases):
    for phase in phases:
        plt.plot(history[phase])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('history.png')

def train(model, dataloader, criterion, optimizer, num_epochs, device='cpu'):
    since = time.time()
    
    acc_history = {
        'train': [],
        'valid': [],
    }

    # Early stopping
    es_patience = 10
    es_trigger = 0
    es_threshold = 0.01

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    pre_loss = 0.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        phases = ['train', 'valid']

        for phase in phases:
        
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            print('{} | Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # if epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     checkpoint = model.state_dict()

            if phase == 'valid' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                print()

            acc_history[phase].append(epoch_acc.to('cpu'))

        # Early stopping
        if np.abs(epoch_loss - pre_loss) < es_threshold:
            es_trigger += 1
            if es_trigger >= es_patience:
                print("Early Stoppying.")
                break
        else:
            es_trigger = 0
                
        #     trigger += 1
        #     if trigger >= patience:
        #         print("Early stopping!")
        #         return model.state_dict()
        pre_loss = epoch_loss
        # last_loss = valid_loss
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation epoch: {}'.format(best_epoch))
    print('Best validation accuracy: {:4f}'.format(best_acc))

    return best_model_wts, acc_history

def predict(model, device):
    with open(input_path + 'testing_img_order.txt') as f:
        test_images_paths = [x.strip() for x in f.readlines()]  # all the testing images
    
    submission = []
    for image_f in test_images_paths:
        image = Image.open(input_path + 'test/' + image_f)
        batch_image = torch.stack([data_transforms['test'](image).to(device)])
        predict = model(batch_image)
        pred_prob = torch.argmax(predict).tolist()
        submission.append([image_f, classes[pred_prob]])

    np.savetxt('answer.txt', submission, fmt='%s')
    return

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='load pretrained model state file')
    parser.add_argument('-o', '--output', type=str, default='best.ckpt', help='output model state file name')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size.')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='number of training epoches.')
    return parser

def main(args):    
    train_set = birdDataset(datatype='train', datalist='training_labels.txt')
    train_set_size = int(len(train_set) * 0.9)
    valid_set_size = len(train_set) - train_set_size

    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])

    print("Training data: %d\nTesting data: %d" % (len(train_set), len(valid_set)))
    print()
    
    dataloaders = {
        'train': DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True, num_workers=4),
        'valid': DataLoader(dataset=valid_set, batch_size=args.batch, shuffle=True, num_workers=4),
    }

    # Select GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Show image
    # for images, labels in dataloaders['train']:
    #     # print("Image:", images)
    #     # print('Label:', labels)
    #     imshow(torchvision.utils.make_grid(images))
    #     print(' '.join('%5s' % classes[int(labels[j])] for j in range(len(labels))))
    #     break

    model = models.resnet152(pretrained=True).to(device)

    # for param in model.fc.parameters():
    #     param.requires_grad = False
    
    # model.fc = nn.Linear(2048, len(classes)).to(device)
    model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, len(classes))).to(device)

    if args.input:
        model.load_state_dict(torch.load(args.input))
    else:
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        best_ckpt, acc_history = train(model, dataloaders, criterion, optimizer, num_epochs=args.epoch, device=device)
        torch.save(best_ckpt, args.output)
        
        model.load_state_dict(best_ckpt)
        show_acc(acc_history, ['train', 'valid'])

    print("Finish model training.")
    print()
    

    # model.eval()
    # print('Start predict...')
    # predict(model, device=device)
    # print("Finish!")

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    main(args)
