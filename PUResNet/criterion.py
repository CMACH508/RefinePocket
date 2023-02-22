# _date_:2021/8/27 17:58

import torch.nn as nn
import torch


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.smoothing_factor = 0.01
    def forward(self, y_true, y_pred):
        bs = y_true.shape[0]
        y_true_f = y_true.view(bs, -1)
        y_pred_f = y_pred.view(bs, -1)
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection + self.smoothing_factor) / (y_true_f.sum() + y_pred_f.sum() + self.smoothing_factor)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice = Dice()

    def forward(self, y_true, y_pred):
        score = 1 - self.dice(y_true, y_pred)
        return score


class Ovl(nn.Module):
    def __init__(self):
        super(Ovl, self).__init__()
        self.smoothing_factor=0.01

    def forward(self, y_true, y_pred):
        concat = torch.cat([y_true, y_pred], dim=1)
        # print(concat.shape) # [100, 2, 36, 36, 36]
        # print(torch.min(concat, dim=1)[0].shape)
        return ((torch.min(concat, dim=1)[0]).sum() + self.smoothing_factor) / ((torch.max(concat, dim=1)[0]).sum() + self.smoothing_factor)


def dice(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    keras layers.
    """
    bs = y_true.shape[0]
    y_true_f = y_true.view(bs, -1)
    y_pred_f = y_pred.view(bs, -1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smoothing_factor) / (y_true_f.sum() + y_pred_f.sum() + smoothing_factor)


def dice_loss(y_true, y_pred, smoothing_factor=0.01):
    """Keras loss function for Dice coefficient (loss(t, y) = -dice(t, y))"""
    # bs = y_true.shape[0]
    # y_true_f = y_true.view(bs, -1)
    # y_pred_f = y_pred.view(bs, -1)
    # intersection = (y_true_f * y_pred_f).sum()
    # score = 2. * (intersection.sum(1) + smoothing_factor) / (y_true_f.sum(1) + y_pred_f.sum(1) + smoothing_factor)
    # score = 1 - score.sum() / bs
    # return - dice(y_true, y_pred)
    score = 1 - dice(y_true, y_pred)
    return score


def ovl(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with keras layers"""
    concat = torch.cat([y_true, y_pred], dim=1)
    # print(concat.shape) # [100, 2, 36, 36, 36]
    # print(torch.min(concat, dim=1)[0].shape)
    return ((torch.min(concat, dim=1)[0]).sum() + smoothing_factor) / ((torch.max(concat, dim=1)[0]).sum() + smoothing_factor)
