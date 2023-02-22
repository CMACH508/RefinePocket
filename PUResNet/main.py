# _date_:2021/8/27 13:25
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn
import argparse
from os.path import join
import time

torch.backends.cudnn.bencmark = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(101)

from ResNet36 import PUResNet
import dataset
from dataset import TrainscPDB
from criterion import Ovl, Dice, DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='')
parser.add_argument('--batch_size', type=int, default=5, help='')
# parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--test_batch_size', type=int, default=5, help='')
# parser.add_argument('--base_lr', type=float, default=1e-5, help='')
parser.add_argument('--base_lr', type=float, default=1e-3, help='')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
parser.add_argument('--lr_adjust', nargs='+', type=int, default=[20000], help='')
parser.add_argument('--max_epoch', type=int, default=1050, help='')
parser.add_argument('--print_freq', type=int, default=300, help='')
parser.add_argument('--save_dir', type=str, default=None, help='')
parser.add_argument('--data_root', type=str, default=None, help='')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.save_dir is None:
    # args.save_dir = 'test_output_' + time.strftime('%Y-%m-%d')
    args.save_dir = 'del_test'
    # args.save_dir = 'test_vnet'
    # args.save_dir = 'baseline_1e-5'
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)



def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def validate(model, device, dice_loss):
    val_dataset = TrainscPDB(subset='validation')
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True, num_workers=10, pin_memory=True)
    model.eval()
    loss_list = []
    with torch.no_grad():
        for ite, (protien, label) in enumerate(val_loader, start=1):
            protien, label = protien.to(device), label.to(device)
            predy = model(protien)
            tot_loss = dice_loss(y_pred=predy, y_true=label)
            tot_loss = tot_loss.item()
            loss_list.append(tot_loss)
    return np.mean(loss_list)


def main():
    dataset.DATA_ROOT = args.data_root
    device = torch.device('cuda')

    model = PUResNet().to(device)

    model = DataParallel(model)
    bce = nn.BCELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-4)  # l2_lambda

    '''lr如何衰减 to do'''
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_adjust, gamma=0.1)

    train_dataset = TrainscPDB(subset='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=12, pin_memory=True)
    min_val_loss = 100
    start_epo = 0

    resume = False
    if resume:
        start_epo = 1050
        args.max_epoch = 1500
        checkpoint = torch.load('')
        model.load_state_dict(checkpoint)
        args.save_dir = ''
        print('-------- load model successfully --------')
        min_val_loss = 0

    dice_loss = DiceLoss().to(device)
    dice = Dice().to(device)
    ovl = Ovl().to(device)

    # old_protein, old_label = None, None
    # val_loss = validate(model, device, dice_loss)
    # print('val_loss={:.4f}'.format(val_loss))
    for epo in range(start_epo, args.max_epoch):
        scheduler.step()
        model.train()
        a_time = datetime.datetime.now()
        print('train_loader=', len(train_loader))
        for ite, (protein, label) in enumerate(train_loader, start=0):
            # if ite == 10:
            #     break
            # print('protien:', protein.shape)  # [b, 18, 36, 36, 36]
            # print('label:', label.shape)  # [b, 1, 36, 36, 36]
            protein, label = protein.to(device), label.to(device)

            # if ite == 0 and epo == 0:
            #     old_protein, old_label = protein, label
            # TODO delete
            # protein, label = old_protein, old_label

            predy = model(protein)
            # print(predy.shape)  # [10, 1, 36, 36, 36]

            tot_loss = dice_loss(y_pred=predy, y_true=label)
            optimizer.zero_grad()
            tot_loss.backward()
            tot_loss = tot_loss.item()
            optimizer.step()

            if ite % 400 == 0:
            # if ite % 1 == 0:
                metric_dice = dice(y_pred=predy, y_true=label).item()
                metric_ovl = ovl(y_pred=predy, y_true=label).item()
                metric_bce = bce(predy, label).item()

                print('epoch : %2d|%2d, iter:%3d|%3d,loss:%.4f,dice:%.3f,ovl:%.3f,bce:%.3f,lr={%.6f}' %
                      (epo, args.max_epoch, ite, len(train_loader), tot_loss, metric_dice, metric_ovl, metric_bce,
                       optimizer.param_groups[0]['lr']))
        b_time = datetime.datetime.now()

        if epo % 20 == 0:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(epo)))

        val_loss = validate(model, device, dice_loss)
        print('val_loss={:.4f}'.format(val_loss))
        c_time = datetime.datetime.now()
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print('------- sota found, epo={} loss={}'.format(epo, val_loss))
        print(get_time(), 'train:{}s val:{}s'.format((b_time - a_time).seconds, (c_time - b_time).seconds))

    torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(epo)))


if __name__ == '__main__':
    print('start time:', get_time())
    main()
    print('end time:', get_time())
