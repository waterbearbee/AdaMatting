import os
import torch
import torchvision
from utility import get_args, get_logger
from tensorboardX import SummaryWriter

from net.adamatting import AdaMatting
from loss import task_uncertainty_loss
print(torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
def train(model, args):
    model.train()

def test():
    pass

def main():
    args = get_args()
    logger = get_logger(args.log)

    device_ids = args.gpu.split(',')
    device_ids = list(map(int, device_ids))

    if args.mode != "train" and args.mode != "test":
        logger.info("Invalid mode. Set mode to \'train\' or \'test\'")
        exit()

    logger.info("Loading network")
    model = AdaMatting(in_channel=4)
    if args.cuda:
        model = model.cuda(device=device_ids[0])
        if len(device_ids) > 1 and args.mode=="train":
            logger.info("Loading with multiple GPUs")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info("Network Loaded")

    model(torch.rand([2, 4, 320, 320], device="cuda"))

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train(model=model, \
            args=args
        )
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test()

if __name__ == "__main__":
    main()