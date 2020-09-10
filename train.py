# -*- coding:utf-8 -*-
import time
import argparse
import random
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import settings
from models import MobileNet
from data import load_data
from utils import str2bool, checkdir, get_dataloader, WarmUpLR, save_train_ckpt, load_train_ckpt, get_newest_file

BASE_DIR = osp.dirname(osp.abspath(__file__))

def train(args) :
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkdir(osp.join(BASE_DIR, settings.checkpoint_dir))
    checkdir(osp.join(BASE_DIR, settings.result_dir))

    train_dataset = load_data.FruitDataset(mode='train')
    test_dataset = load_data.FruitDataset(mode='test')
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers, True)
    test_loader = get_dataloader(test_dataset, args.batch_size, args.num_workers, False)

    net = MobileNet.get_model(settings.num_classes).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                     gamma=0.2)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    ckpt_path = osp.join(BASE_DIR, "checkpoint")

    start_epoch = -1
    train_infos = dict()
    train_infos['loss'] = list()
    if args.resume :
        try :
            start_epoch, pre_infos = load_train_ckpt([net, optimizer], get_newest_file(ckpt_path), device)
            train_infos['loss'] += pre_infos
        except :
            print("Fail to load checkpoints...")

    for epoch in range(start_epoch + 1, args.epochs) :
        net.train()
        if epoch > args.warm:
            train_scheduler.step(epoch)

        start = time.time()
        train_loss, train_corrects = 0, 0

        for iter_idx, (img, label) in  enumerate(train_loader) :
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            predict = net(img)
            loss = loss_function(predict, label)
            train_loss += loss.item()
            loss.backward()
            predict = nn.functional.softmax(predict, dim=1)
            _, preds = predict.max(1)
            train_corrects += preds.eq(label).sum()
            optimizer.step()

            if epoch <= args.warm:
                warmup_scheduler.step(epoch)

        finish = time.time()
        train_infos['loss'].append(train_loss / len(train_loader.dataset))
        print('epoch {} : Average loss: {:.4f}, Accuracy: {:.2%}, Time consumed:{:.2f}s'.format(
            epoch,
            train_loss / len(train_loader.dataset),
            float(train_corrects) / len(train_loader.dataset),
            finish - start
        ))

        if epoch % 2 :
            net.eval()
            start = time.time()
            test_corrects, test_loss = 0, 0
            for iter_idx, (img, label) in enumerate(test_loader) :
                img, label = img.to(device), label.to(device)
                predict = net(img)
                loss = loss_function(predict, label)
                test_loss += loss.item()
                predict = nn.functional.softmax(predict, dim=1)
                _, preds = predict.max(1)
                test_corrects += preds.eq(label).sum()
            finish = time.time()
            print('Evaluating Network.....')
            print('Test set: Average loss: {:.4f}, Accuracy: {:.2%}, Time consumed:{:.2f}s'.format(
                test_loss / len(test_loader.dataset),
                float(test_corrects) / len(test_loader.dataset),
                finish - start
            ))

        if epoch % 2 :
            save_train_ckpt([net, optimizer], epoch, train_infos,
                            osp.join(BASE_DIR, "checkpoint", "epoch_{:d}_acc_{:.2%}.ckpt".format(epoch, float(train_corrects) / len(train_loader.dataset))))

    torch.save(net.state_dict(), osp.join(BASE_DIR, "result", "weights.pt"))

    color = (random.random(), random.random(), random.random())
    plt.plot([i for i in range(len(train_infos['loss']))], train_infos['loss'], label="train", color=color, marker='.')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(osp.join(BASE_DIR, "result", "loss.png"), dpi=300, bbox_inches='tight')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='PyTorch Fruit Recognition')
    parser.add_argument('--epochs', '-e', default=160, type=int, help='the totoal number of epochs when training')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--num_workers', '-nw', type=int, default=2,
                        help='how many subprocesses to use for data loading')
    parser.add_argument('--num_classes', '-nc', type=int, default=30, help='the number of classes')
    parser.add_argument('--resume', '-r', type=str2bool, default='t',
                        help='resume from checkpoint')
    parser.add_argument('--milestones', '-m', type=list, default=[60, 120, 160], help='the interval between changing learning rate')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    args = parser.parse_args()

    train(args)