import os
import torch
import torchvision
from utility import get_args, get_logger
from net.adamatting import AdaMatting
from torchsummary import summary

def train():
    pass

def test():
    pass

def main():
    args = get_args()
    logger = get_logger(args.log)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cuda:
        if torch.cuda.is_available():
            logger.info("Running with GPUs: %s" % args.gpu)
        else:
            raise Exception("No GPU found, please run without --cuda")
    else:
        logger.info("Running without GPU")

    if args.mode != "train" and args.mode != "test":
        logger.info("Invalid mode. Set mode to \'train\' or \'test\'")
        exit()

    logger.info("Loading network")
    model = AdaMatting(in_channel=4)
    model = model.cuda()
    logger.info("Network Loaded")
    if args.debug:
        summary(model, (4, 320, 320))

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train()
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test()

if __name__ == "__main__":
    main()