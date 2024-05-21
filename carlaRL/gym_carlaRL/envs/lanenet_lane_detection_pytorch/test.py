import time
import os
import sys

import torch
from model.lanenet.LaneNet import LaneNet
# from dataloader.transformers import Rescale
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
# import pandas as pd
import cv2


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = './test_output/input.jpg'
    resize_height = 256
    resize_width = 512

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = './log/loss=0.1223_miou=0.5764_epoch=73.pth'
    model = LaneNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    Time_start = time.time()
    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    Time_preprocess = time.time()
    print('Preprocess Time: ', Time_preprocess - Time_start)
    with torch.no_grad():
        outputs = model(dummy_input)
    Time_inference = time.time()
    print('LaneNet Inference Time: ', Time_inference - Time_preprocess)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)


if __name__ == "__main__":
    test()