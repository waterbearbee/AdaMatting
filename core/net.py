import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F
import convlstm


class AdaMatting(nn.Module):
    def __init__(self):
        super(AdaMatting, self).__init__()
        