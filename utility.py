import argparse
import logging


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='set arguments')
    parser.add_argument('--mode', type=str, required=True, help="set the program to \'train\' or \'test\'")
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=str, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=str, required=True, help="crop width size of input image")
    parser.add_argument('--alphaDir', type=str, required=True, help="directory of alpha")
    parser.add_argument('--fgDir', type=str, required=True, help="directory of fg")
    parser.add_argument('--bgDir', type=str, required=True, help="directory of bg")
    parser.add_argument('--imgDir', type=str, default="", help="directory of img")
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=10, help='epoch of learning decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--gpu', type=str, default="0", help="choose gpus")
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    parser.add_argument('--saveDir', type=str, help="checkpoint that model save to")
    parser.add_argument('--printFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--ckptSaveFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--wl_weight', type=float, default=0.5, help="alpha loss weight")
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2, 3], help="training stage: 0(simple loss), 1, 2, 3")
    parser.add_argument('--testFreq', type=int, default=-1, help="test frequency")
    parser.add_argument('--testImgDir', type=str, default='', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='', help="test result save to")
    parser.add_argument('--crop_or_resize', type=str, default="whole", choices=["resize", "crop", "whole"], help="how manipulate image before test")
    parser.add_argument('--max_size', type=int, default=1312, help="max size of test image")
    parser.add_argument('--log', action="store_true", default=False, help="whether store log to log.txt")
    parser.add_argument('--arch', type=str, default='vgg', help="network structure")
    args = parser.parse_args()
    return args
    

def get_logger(flag):
    logger = logging.getLogger("AdaMatting")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s %(lineno)d: %(levelname)s - %(message)s")

    # log file stream
    if (flag):
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger
