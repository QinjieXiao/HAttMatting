import io
import os
import pickle
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from torchsummaryX import summary
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from model import E2EModel, Model
from utils import AverageMeter, draw_str

# from large_model import Model

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix)+1:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key] 
    # first_state_name = list(state_dict.keys())[0]
    # if not first_state_name.startswith(prefix):
    #     for key, value in state_dict.items():
    #         new_state_dict[key] = state_dict[key].float()
    # else:
    #     for key, value in state_dict.items():
    #         new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict

def has_prefix_state_dict(state_dict, prefix='module'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def load_state_dict(model, state_dict):
    new_state_dict = {}
    for name, p in model.named_parameters():
        if name in state_dict:
            new_state_dict[name] = state_dict[name]
    model.load_state_dict(new_state_dict)
    return model
def my_torch_load(fname): 
    try: 
        ckpt = torch.load(fname, map_location=lambda storage, loc: storage) 
        return ckpt 
    except Exception as e: 
        print("Load Error:{}\nTry Load Again...".format(e)) 
        class C: 
            pass 
        def c_load(ss, encoding='latin1'): 
            return pickle.load(ss, encoding='latin1') 
        def c_unpickler(ss, encoding='latin1'): 
            return pickle.Unpickler(ss, encoding='latin1') 
        c = C 
        c.load = c_load
        c.Unpickler = c_unpickler 
        ckpt = torch.load(fname, pickle_module=c, map_location=lambda storage, loc: storage) 
        return ckpt 

def SAD(trimap_pred, trimap_true):
    # multiply with 10 to 
    return np.mean(np.absolute(trimap_pred - trimap_true)) * 10

def MSE(trimap_pred, trimap_true):
    return np.mean(np.square(trimap_pred - trimap_true))
    
if __name__ == '__main__':
    myModel = Model('train_trimap', attention=True)
    without_gpu = True
    device = 'cpu'
    
    model = './checkpoint-epoch=15-val_loss=0.1163.ckpt'
    print('Loading model from {}...'.format(model))
    if without_gpu:
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model)
    myModel.load_state_dict(checkpoint['state_dict'])

    # model = './ckpt_lastest.pth'
    # print('Loading model from {}...'.format(model))
    # if without_gpu:
    #     checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
    # else:
    #     checkpoint = torch.load(model)
    # myModel.trimap.migrate(remove_prefix_state_dict(checkpoint['state_dict'], 't_net'))

    model = './gca-dist.pth'
    print('Loading model from {}...'.format(model))
    if without_gpu:
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model)
    myModel.alpha.migrate(remove_prefix_state_dict(checkpoint['state_dict'], 'module'))


    myModel.eval()
    myModel.to(device)

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    for img_name in tqdm(os.listdir(os.path.join('..', 'dataset', 'training', 'image'))):
        img_path = os.path.join('..', 'dataset', 'training', 'image', img_name)
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,12:-12]

        tm_name = img_name[:-4] + '_trimap' + img_name[-4:]
        tm_path = os.path.join('..', 'dataset', 'training', 'trimap', tm_name)
        print(tm_path)
        tm = cv2.imread(tm_path, cv2.IMREAD_UNCHANGED)
        tm = tm[:, 12:-12]
        
        # img = data_transforms['valid'](img)
        # img = img.unsqueeze(0)
        # img = img / 255.0
        trimap, alpha = myModel(img)

        trimap = trimap.argmax(dim=1, keepdim=True)
        trimap = trimap.squeeze(0).squeeze(0).detach()
        # trimap[trimap == 0] = 0
        trimap[trimap == 1] = 128
        trimap[trimap == 2] = 255
        trimap = trimap.numpy().astype(np.uint8)

        alpha = alpha.squeeze(0).squeeze(0).detach()
        alpha[trimap <= 0] = 0
        alpha[trimap >= 255] = 1
        alpha = (alpha * 255).numpy().astype(np.uint8)

        # cv2.imshow('trimap', trimap)
        # cv2.imshow('alpha', alpha)
        # cv2.imshow('trimap_gt', tm)
        mse_loss = MSE(trimap, tm)
        sad_loss = SAD(trimap, tm)
        mse_losses.update(mse_loss)
        sad_losses.update(sad_loss)
        print("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        draw_str(alpha, (10, 20), "sad:{} mse:{}".format(
            sad_loss.item(), mse_loss.item()))
        out_path = os.path.join('result', img_name)
        cv2.imwrite(out_path, alpha)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# if __name__ == "__main__":
#     model = Model('train_trimap', attention=False)
#     # print(model.state_dict().keys())
#     checkpoint = torch.load('checkpoint-epoch=15-val_loss=0.1163.ckpt', map_location=lambda storage, loc: storage)
#     print(checkpoint['state_dict'].keys())
