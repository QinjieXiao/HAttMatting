from tqdm import tqdm
from time import time

import torch
import torchvision
from torchsummary import summary
# from torchsummaryX import summary
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import io
import pickle
from large_model import Model

# from large_model import Model
 
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
    # checkpoint = my_torch_load('stage1_skip_sad_52.9.pth')
    # print(checkpoint['state_dict'].keys())
    model = Model('train_trimap')
    summary(model, (3, 800, 576))
    # with open('stage1_skip_sad_52.9.pth', 'rb') as f:
    #     buffer = io.BytesIO(f.read())
    # torch.load(buffer, map_location=lambda storage, loc: storage, encoding='utf8')
    # print(checkpoint)
    # model = Model('train_trimap')
    # summary(model, (3, 800, 600))
    # summary(model, (3, 322, 322))
    # model.migrate()
    # inp = torch.zeros((2, 3, 320, 320))
    # out1, out2 = model(inp)
    # print(out1.shape, out2.shape)
    # summary(model, torch.zeros((2, 3, 320, 320)))