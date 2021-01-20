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
    model = E2EModel()
    checkpoint = torch.load('model_r34_2b_gfm_tt.pth', map_location=lambda storage, loc: storage)
    # print(checkpoint.keys())
    model.migrate(checkpoint)
    # model = Model('train_trimap')
    # model.eval()
    # checkpoint = remove_prefix_state_dict(torch.load('ckpt_lastest.pth', map_location=lambda storage, loc: storage)['state_dict'], 't_net')
    # # checkpoint = torch.load('checkpoint-epoch=694-val_loss=0.1661.ckpt', map_location=lambda storage, loc: storage)['state_dict']
    # model.trimap.migrate(checkpoint)
    # checkpoint = remove_prefix_state_dict(torch.load('gca-dist.pth', map_location=lambda storage, loc: storage)['state_dict'], 'module')
    # model.alpha.migrate(checkpoint)
    for img_name in tqdm(os.listdir(os.path.join('dataset', 'testing', 'image'))):
        img_path = os.path.join('dataset', 'testing', 'image', img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,12:-12]
        img = data_transforms['valid'](img)
        img = img.unsqueeze(0)
        img = img / 255.0
        _, _, alpha = model(img)

        # trimap = trimap.squeeze(0).squeeze(0).detach()
        # trimap[trimap == 0] = 0
        # trimap[trimap == 1] = 128
        # trimap[trimap == 2] = 255
        # trimap = trimap.numpy().astype(np.uint8)

        alpha = alpha.squeeze(0).squeeze(0).detach()
        # alpha[trimap <= 0] = 0
        # alpha[trimap >= 255] = 1
        alpha = (alpha * 255).numpy().astype(np.uint8)

        cv2.imwrite(os.path.join('dataset', 'testing', 'pred_large', img_name), alpha)

        # trimap = trimap[..., np.newaxis]
        # alpha = alpha[..., np.newaxis]
        # cv2.imshow('trimap', np.concatenate([trimap, trimap, trimap], axis=2).astype(np.uint8))
        # cv2.imshow('alpha', np.concatenate([alpha, alpha, alpha], axis=2).astype(np.uint8))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        # cv2.destroyAllWindows()

        # cv2.imshow()
        # plt.imshow(alpha, cmap='gray')
        # plt.show()
        # plt.imshow(trimap, cmap='gray')
        # plt.show()