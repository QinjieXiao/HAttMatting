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

# from large_model import Model

if __name__ == '__main__':
    # checkpoint = torch.load('stage1_skip_sad_52.9.pth', map_location=lambda storage, loc: storage, encoding='ascii')
    with open('stage1_skip_sad_52.9.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())
    torch.load(buffer, map_location=lambda storage, loc: storage, encoding='utf8')
    print(checkpoint)
    # model = Model('train_trimap')
    # summary(model, (3, 800, 600))
    # summary(model, (3, 322, 322))
    # model.migrate()
    # inp = torch.zeros((2, 3, 320, 320))
    # out1, out2 = model(inp)
    # print(out1.shape, out2.shape)
    # summary(model, torch.zeros((2, 3, 320, 320)))