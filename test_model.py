from tqdm import tqdm
from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary
# from torchsummaryX import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import io
import pickle
from model import Model, E2EModel

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
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

if __name__ == '__main__':
    myModel = Model('train_trimap')
    without_gpu = True
    device = 'cpu'

    model = './ckpt_lastest.pth'
    print('Loading model from {}...'.format(model))
    if without_gpu:
        checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model)
    myModel.trimap.migrate(remove_prefix_state_dict(checkpoint['state_dict'], 't_net'))

    model = './stage1_skip_sad_52.9.pth'
    print('Loading model from {}...'.format(model))
    if without_gpu:
        checkpoint = my_torch_load(model)
    else:
        checkpoint = my_torch_load(model)
    myModel.alpha.migrate(checkpoint['state_dict'])
    # if args.without_gpu:
    #     myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    # else:
    #     myModel = torch.load(args.model)

    myModel.eval()
    myModel.to(device)

    for img_name in tqdm(os.listdir(os.path.join('..', 'dataset', 'testing', 'image'))):
        img_path = os.path.join('..', 'dataset', 'testing', 'image', img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,12:-12]
        
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

        cv2.imshow('trimap', trimap)
        cv2.imshow('alpha', alpha)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
