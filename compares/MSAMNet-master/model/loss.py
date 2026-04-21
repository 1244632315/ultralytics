import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F

def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = 1
        else:
            self.alpha = alpha
            self.size_average = size_average

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CenterLoss(nn.Module):
    def __init__(self, lambda_hm=1, lambda_off=1, lambda_size=0.1):
        super(CenterLoss, self).__init__()
        self.l1 = lambda_hm
        self.l2 = lambda_size
        self.l3 = lambda_off

    def forward(self, preds, targets):
        pred_heatmap, pred_size, pred_offset = preds
        gt_heatmap, gt_offset, gt_size, mask = targets
        
        self.loss_hm = self.focal_loss(pred_heatmap, gt_heatmap, gamma=1.5, alpha=1, beta=5)
        self.loss_off = self.l1_loss(pred_offset, gt_offset, mask)
        self.loss_size = self.l1_loss(pred_size, gt_size, mask)
        
        total_loss = self.l1 * self.loss_hm + self.l3 * self.loss_off + self.l2 * self.loss_size
        return total_loss

    def focal_loss(self, pred, target, gamma=2, alpha=1.5, beta=4, posthresh=0.8, smallweight=2):
        # gamma: hard examples weight param
        # beta: non-center punish weight param
        # alpha: positive examples weight param
        pos_inds = target.ge(posthresh).float()
        neg_inds = target.lt(posthresh).float()

        # The negative samples near the positive sample feature point have smaller weights
        neg_weights = torch.pow(1-target, beta)
        loss = 0
        pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

        weight = torch.ones_like(target)
        weight[target > 0.5] = smallweight

        # Calculate Focal Loss.
        # The hard to classify sample weight is large, easy to classify sample weight is small.
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds * weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_inds * neg_weights * weight

        # Loss normalization is carried out
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum() * alpha
        neg_loss = neg_loss.sum() 
        # print('pos loss: %f, neg loss: %f'%(pos_loss, neg_loss))

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
    
    def l1_loss(self, pred, target, mask):
        expand_mask = torch.unsqueeze(mask, 1).repeat(1, 2, 1, 1)

        # Don't calculate loss in the position without ground truth.
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-7)

        return loss
