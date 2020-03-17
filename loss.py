import torch.nn as nn

def trimap_adaptation_loss(args, pred_trimap, gt_trimap):
    loss = nn.CrossEntropyLoss()

def alpha_estimation_loss(args, pred_alpha, gt_alpha):
    pass

def task_uncertainty_loss(args, pred_trimap, pred_alpha, gt_trimap, gt_alpha, sigma_t, sigma_a):
    pass
