import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import operator
from einops import rearrange
from medpy import metric
import SimpleITK as sitk


"""
该文件包括 ：
（1）训练需要的 dice_loss dice
（2）评估需要的 specificity sensitivity Hausdorff_distance
"""
import torch


def hausdorff_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the Hausdorff distance between two sets of points x and y.

    Arguments:
    x -- a tensor of shape (n_x, d), representing the first set of points
    y -- a tensor of shape (n_y, d), representing the second set of points

    Returns:
    The Hausdorff distance between the two sets of points
    """
    n_x, n_y = x.shape[0], y.shape[0]
    assert x.shape[1] == y.shape[1], "Input dimensions must match"

    # Compute pairwise distances between all points
    dist_matrix = torch.cdist(x.float(), y.float())

    # Compute the minimum distance from each point in x to y
    min_dist_x = torch.min(dist_matrix, dim=1)[0]

    # Compute the minimum distance from each point in y to x
    min_dist_y = torch.min(dist_matrix, dim=0)[0]

    # Return the maximum of the two minimum distances
    return torch.max(torch.stack([min_dist_x.max(), min_dist_y.max()]))


def hausdorff_distance_WT(pre, label):
    pre = torch.cat([pre[0, 1, ...], pre[0, 2, ...], pre[0, 3, ...]], dim=1)
    label = torch.cat([label[0, 1, ...], label[0, 2, ...], label[0, 3, ...]], dim=1)
    return hausdorff_distance(pre, label)


def hausdorff_distance_ET(pre, label):
    pre = pre[0, 3, ...]
    label = label[0, 3, ...]
    return hausdorff_distance(pre, label)


def hausdorff_distance_TC(pre, label):
    pre = torch.cat([pre[0, 1, ...], pre[0, 3, ...]], dim=1)
    label = torch.cat([label[0, 1, ...], label[0, 3, ...]], dim=1)
    return hausdorff_distance(pre, label)


def sensitivity(predict, target):
    result = metric.binary.sensitivity(predict, target)
    return result
#
# def sensitivity(output, target):
#     """Compute sensitivity of a binary classifier."""
#     # Convert output to binary predictions (0 or 1)
#     output, target = torch.from_numpy(output), torch.from_numpy(target)
#     preds = torch.round(torch.sigmoid(output))
#     # Find indices of positive and negative examples in target
#     pos_idx = (target == 1).nonzero()
#     neg_idx = (target == 0).nonzero()
#     # Calculate true positives (TP) and false negatives (FN)
#     TP = torch.sum(preds[pos_idx] == target[pos_idx])
#     FN = torch.sum(preds[neg_idx] != target[neg_idx])
#     # Calculate sensitivity (TPR = TP / (TP + FN))
#     sensitivity = TP / (TP + FN)
#     return sensitivity


def sensitivity_WT(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 2, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 2, ...], target[:, 3, ...]], axis=1)
    result = sensitivity(predict, target)

    return result


def sensitivity_ET(predict, target):
    predict = predict[:, 3, ...]
    target = target[:, 3, ...]

    result = sensitivity(predict, target)

    return result


def sensitivity_TC(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 3, ...]], axis=1)

    result = sensitivity(predict, target)

    return result

def specificity(predict, target):
    result = metric.binary.specificity(predict, target)
    return result
#
# def specificity(y_true, y_pred):
#     # # 将预测值 y_pred 转换为类别概率
#     # y_pred = torch.softmax(y_pred, dim=1)
#     # # 获取预测结果中概率最大的类别
#     # y_pred = torch.argmax(y_pred, dim=1)
#     y_true, y_pred = torch.from_numpy(y_true), torch.from_numpy(y_pred)
#     # 计算真实标签和预测标签不同的位置
#     mask = (y_true != y_pred).float()
#     # 计算真实标签中负样本的数量
#     tn = torch.sum((y_true == 0).float() * (mask == 1))
#     # 计算真实标签中所有负样本的数量
#     total_negatives = torch.sum((y_true == 0).float())
#     # 计算 specificity
#     spec = tn / (total_negatives + 1e-7)
#     return spec


def specificity_WT(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 2, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 2, ...], target[:, 3, ...]], axis=1)
    result = specificity(predict, target)

    return result


def specificity_ET(predict, target):
    predict = predict[:, 3, ...]
    target = target[:, 3, ...]

    result = specificity(predict, target)

    return result


def specificity_TC(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 3, ...]], axis=1)

    result = specificity(predict, target)

    return result


def Dice(pred, target):
    """计算3D数据的Dice系数"""
    smooth = 1.0

    # 将预测和目标张量展平成一维向量
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    # 计算并集
    pred_sum = torch.sum(pred_flat)
    target_sum = torch.sum(target_flat)
    union = pred_sum + target_sum

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET): label4
    dice2(TC): label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        # input: (1, 4, 160, 160, 64)
        # target: (1, 160, 160, 64)
        smooth = 1e-6

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target, self.n_classes)
        # input1: torch.Size([1, 4, 160, 160, 64])
        # target1: torch.Size([1, 160, 160, 64, 4])

        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')
        # input1: torch.Size([1, 4, (160*160*64)])
        # target1: torch.Size([1, 4, (160*160*64)])

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()
        # input1: torch.Size([1, 3, (160*160*64)])
        # target1: torch.Size([1, 3, (160*160*64)])

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))

    x = torch.ones((1, 4, 2, 2, 2))
    y = torch.ones((1, 4, 2, 2, 2))
    out = hausdorff_distance(x, y)
    print(out)
