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

data_transforms = transforms.Compose([
                    transforms.Resize((244,244)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    ])

def predict(model, device):
    with open(input_path + 'testing_img_order.txt') as f:
        test_images_paths = [x.strip() for x in f.readlines()]  # all the testing images
    
    submission = []
    for image_f in test_images_paths:
        image = Image.open(input_path + 'test/' + image_f)
        batch_image = torch.stack([data_transforms(image).to(device)])
        predict = model(batch_image)
        pred_prob = torch.argmax(predict).tolist()
        submission.append([image_f, classes[pred_prob]])

    np.savetxt('answer.txt', submission, fmt='%s')
    return

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='load model state file', required=True)
    return parser

def main(args):    
    # Select GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.resnet152(pretrained=True).to(device)
    
    # model.fc = nn.Linear(2048, len(classes)).to(device)
    model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, len(classes))).to(device)

    model.load_state_dict(torch.load(args.input))

    model.eval()

    print('Start predict...')
    predict(model, device=device)
    print("Finish!")

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    main(args)
