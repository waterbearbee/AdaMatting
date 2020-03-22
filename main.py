import os
import torch
import torchvision
from utility import get_args, get_logger, poly_lr_scheduler
from tensorboardX import SummaryWriter

from dataset.dataset import AdaMattingDataset
from dataset.pre_process import composite_dataset, gen_train_valid_names
from net.adamatting import AdaMatting
from loss import task_uncertainty_loss

from torchvision import transforms
from PIL import Image
import numpy as np

def train(model, optimizer, device, args, logger):
    model.train()

    logger.info("Initializing data loaders")
    train_dataset = AdaMattingDataset(args.raw_data_path, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=16, pin_memory=True)
    valid_dataset = AdaMattingDataset(args.raw_data_path, 'valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                               num_workers=16, pin_memory=True)

    logger.info("Start training")
    max_iter = 43100 * 0.8 / args.batch_size * args.epochs
    cur_iter = 0
    for epoch in range(args.epochs):
            for index, (img, gt) in enumerate(train_loader):
                poly_lr_scheduler(optimizer=optimizer, init_lr=args.lr, iter=cur_iter, max_iter=max_iter, power=0.9)
                cur_iter += 1

                img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
                gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
                gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

                optimizer.zero_grad()
                trimap_adaption, t_argmax, alpha_estimation, sigma_t, sigma_a = model(img)
                loss = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                             pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                             gt_alpha=gt_alpha, sigma_t=sigma_t, sigma_a=sigma_a)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if index == 10:
                    exit()


def test():
    pass


def main():
    args = get_args()
    logger = get_logger(args.write_log)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids_str = args.gpu.split(',')
    device_ids = []
    for i in range(len(device_ids_str)):
        device_ids.append(i)

    if args.mode != "prep":
        logger.info("Loading network")
        model = AdaMatting(in_channel=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        if args.cuda:
            model = model.cuda(device=device_ids[0])
            device = torch.device("cuda:{}".format(device_ids[0]))
            if len(device_ids) > 1 and args.mode=="train":
                logger.info("Loading with multiple GPUs")
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            device = torch.device("cpu")

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train(model=model, optimizer=optimizer, device=device, args=args, logger=logger)
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test()
    elif args.mode == "prep":
        logger.info("Program runs in prep mode")
        composite_dataset(args.raw_data_path, logger)
        gen_train_valid_names(logger)


if __name__ == "__main__":
    main()
