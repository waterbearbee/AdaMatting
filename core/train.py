import torch
import os
from utility import get_args, get_logger
import net

def main():
    args = get_args()
    logger = get_logger(args.log)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if __name__ == "__main__":
    main()