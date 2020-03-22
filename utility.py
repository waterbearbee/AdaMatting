import argparse
import logging


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='set arguments')
    parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "prep"], help="set the program to \'train\', \'test\', or \'prep\'")
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=str, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=str, required=True, help="crop width size of input image")
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda?')
    parser.add_argument('--gpu', type=str, default="0", help="choose gpus")
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--testImgDir', type=str, default='', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='', help="test result save to")
    parser.add_argument('--crop_or_resize', type=str, default="whole", choices=["resize", "crop", "whole"], help="how manipulate image before test")
    parser.add_argument('--max_size', type=int, default=1312, help="max size of test image")
    parser.add_argument('--write_log', action="store_true", default=False, help="whether store log to log.txt")
    parser.add_argument('--debug', action="store_true", default=False, help="whether print verbose debug info")
    parser.add_argument('--raw_data_path', type=str, default="/data/datasets/im/AdaMatting/", help="dir where datasets are stored")
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
