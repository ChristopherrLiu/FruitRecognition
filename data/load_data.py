# -*- coding:utf-8 -*-
import os.path as osp
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs

BASE_DIR = osp.dirname(osp.abspath(__file__))

train_data_transforms = tfs.Compose([
    tfs.Resize((256, 256)),
    tfs.RandomRotation(30),
    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data_transforms = tfs.Compose([
    tfs.Resize((256, 256)),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FruitDataset(Dataset) :
    """
    class for load dataset
    :param mode : train or test
    :return:
    """
    def __init__(self, mode="train") :
        self.mode = mode
        self.labels2id = json.load(open(osp.join(BASE_DIR, "labels.json")))
        self.img_infos = json.load(open(osp.join(BASE_DIR, mode+ "_" + "imgs.json")))

    def __len__(self) :
        return len(self.img_infos)

    def __getitem__(self, index) :
        img_info = self.img_infos[str(index)]
        img = Image.open(img_info["path"]).convert('RGB')
        if self.mode == 'train' :
            img = train_data_transforms(img)
        else :
            img = test_data_transforms(img)
        label = int(img_info["filename"].split('_')[0])
        return img, label