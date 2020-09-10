# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as tfs

from models import MobileNet
import settings

BASE_DIR = osp.dirname(osp.abspath(__file__))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = MobileNet.get_model(settings.num_classes).to(device)

data_transforms = tfs.Compose([
    tfs.Resize((256, 256)),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_one_image(img:Image.Image) :
    prcoess_image = data_transforms(img).unsqueeze(0)
    states = torch.load(
        osp.join(BASE_DIR, settings.checkpoint_dir, settings.weight_file_name), map_location=device)['state_dicts'][0]
    net.load_state_dict(states)
    net.eval()

    out = net(prcoess_image)
    _, pred_idx = nn.functional.softmax(out, dim=1).max(1)

    return settings.fruit_idx2name[pred_idx.item()]