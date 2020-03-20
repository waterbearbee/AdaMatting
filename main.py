import os
import torch
import torchvision
from utility import get_args, get_logger, composite_dataset
from tensorboardX import SummaryWriter

from net.adamatting import AdaMatting
from loss import task_uncertainty_loss


def train(model, args):
    model.train()
    for _ in range(1000):
        tmp_trimap = (torch.rand([2, 320, 320]) * 3).long().cuda()
        tmp_alpha = torch.rand([2, 1, 320, 320]).cuda()
        trimap_adaption, t_argmax, alpha_estimation, sigma_t, sigma_a = model(torch.randn([2, 4, 320, 320], device="cuda"))
        loss = task_uncertainty_loss(pred_trimap=trimap_adaption, 
                                    pred_trimap_argmax=t_argmax, 
                                    pred_alpha=alpha_estimation, 
                                    gt_trimap=tmp_trimap, 
                                    gt_alpha=tmp_alpha, 
                                    sigma_t=sigma_t, 
                                    sigma_a=sigma_a)
        print(loss)
        # loss.backward()


def test():
    pass


def main():
    args = get_args()
    logger = get_logger(args.log)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids_str = args.gpu.split(',')
    device_ids = []
    for i in range(len(device_ids_str)):
        device_ids.append(i)

    if args.mode != "prep":
        logger.info("Loading network")
        model = AdaMatting(in_channel=4)
        if args.cuda:
            model = model.cuda(device=device_ids[0])
            if len(device_ids) > 1 and args.mode=="train":
                logger.info("Loading with multiple GPUs")
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("Network Loaded")

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train(model=model, args=args)
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test()
    elif args.mode == "prep":
        logger.info("Program runs in prep mode")
        composite_dataset(args.raw_data_path, logger)


if __name__ == "__main__":
    main()
