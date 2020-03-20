import torch
import torch.nn as nn


def trimap_adaptation_loss(pred_trimap, gt_trimap):
    loss = nn.CrossEntropyLoss()
    return loss(pred_trimap, gt_trimap)


def alpha_estimation_loss(pred_alpha, gt_alpha, pred_trimap_argmax):
    """
    pred_trimap_argmax
    0: foreground
    1: unknown
    2: background
    """
    mask = (pred_trimap_argmax == 1)
    mask = mask.float()
    num_unknown_pixel = torch.sum(mask)
    masked_pred_alpha = pred_alpha.mul(mask)
    masked_gt_alpha = gt_alpha.mul(mask)

    loss = nn.L1Loss()
    return loss(masked_pred_alpha, masked_gt_alpha) / (num_unknown_pixel + 1e-8)


def task_uncertainty_loss(pred_trimap, pred_trimap_argmax, pred_alpha, gt_trimap, gt_alpha, sigma_t, sigma_a):
    Lt = trimap_adaptation_loss(pred_trimap, gt_trimap)
    La = alpha_estimation_loss(pred_alpha, gt_alpha, pred_trimap_argmax)
    return Lt / (2 * sigma_t * sigma_t) + La / (sigma_a * sigma_a) + torch.log(2 * sigma_a * sigma_t)
