import torch
import os
from utility import get_args
import net

def main():
    args = get_args()
    print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if __name__ == "__main__":
    main()