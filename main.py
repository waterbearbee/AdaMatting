import torch
import os
from utility import get_args, get_logger
import net.adamatting

def main():
    args = get_args()
    logger = get_logger(args.log)

    if(args.mode == 'train'):
        logger.info("Program runs in train mode")
    else:
        logger.info("Program runs in test mode")
    
    if (args.cuda):
        if torch.cuda.is_available():
            logger.info("Running with GPUs: %s" % args.gpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        else:
            raise Exception("No GPU found, please run without --cuda")
    else:
        logger.info("Running without GPU")
    
    
    


if __name__ == "__main__":
    main()