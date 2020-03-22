import os
import torch
import torchvision
from utility import get_args, get_logger, poly_lr_scheduler, save_checkpoint, AverageMeter
from tensorboardX import SummaryWriter

from dataset.dataset import AdaMattingDataset
from dataset.pre_process import composite_dataset, gen_train_valid_names
from net.adamatting import AdaMatting
from loss import task_uncertainty_loss


def train(model, optimizer, device, args, logger):
    torch.manual_seed(7)
    writer = SummaryWriter()
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
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        for index, (img, gt) in enumerate(train_loader):
            cur_lr = poly_lr_scheduler(optimizer=optimizer, init_lr=args.lr, iter=cur_iter, max_iter=max_iter, power=0.9)

            img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
            gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
            gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

            optimizer.zero_grad()
            trimap_adaption, t_argmax, alpha_estimation, sigma_t, sigma_a = model(img)
            L_overall, L_t, L_a = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                        pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                        gt_alpha=gt_alpha, sigma_t=sigma_t, sigma_a=sigma_a)

            optimizer.zero_grad()
            L_overall.backward()
            optimizer.step()

            if cur_iter % 10 == 0:
                logger.info("Epoch: {:>3d} | Iter: {:>5d}/{} | Loss: {}".format(epoch, index, len(train_loader), L_overall.item()))
                writer.add_scalar("lr", cur_lr, cur_iter)
                writer.add_scalar("loss/L_overall", L_overall.item(), cur_iter)
                writer.add_scalar("loss/L_t", L_t.item(), cur_iter)
                writer.add_scalar("loss/L_a", L_a.item(), cur_iter)
            
            cur_iter += 1
        
        # Validation
        logger.info("Validating after the {}th epoch".format(epoch))
        avg_loss = AverageMeter()
        avg_l_t = AverageMeter()
        avg_l_a = AverageMeter()
        for index, (img, gt) in enumerate(valid_loader):
            img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
            gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
            gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

            trimap_adaption, t_argmax, alpha_estimation, sigma_t, sigma_a = model(img)
            L_overall_valid, L_t_valid, L_a_valid = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                        pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                        gt_alpha=gt_alpha, sigma_t=sigma_t, sigma_a=sigma_a)
            avg_loss.update(L_overall_valid.item())
            avg_l_t.update(L_t_valid.item())
            avg_l_a.update(L_a_valid.item())

            if index == 0:
                trimap_adaption_res = torchvision.utils.make_grid(t_argmax / 2, normalize=True, scale_each=True)
                writer.add_image('valid_image/trimap_adaptation', trimap_adaption_res, cur_iter)
                alpha_estimation_res = torchvision.utils.make_grid(alpha_estimation, normalize=True, scale_each=True)
                writer.add_image('valid_image/alpha_estimation', alpha_estimation_res, cur_iter)

        logger.info("Loss overall: {}".format(L_overall_valid.item()))
        logger.info("Loss of trimap adaptation: {}".format(L_t_valid.item()))
        logger.info("Loss of alpha estimation: {}".format(L_a_valid.item()))
        writer.add_scalar("valid_loss/L_overall", L_overall.item(), cur_iter)
        writer.add_scalar("valid_loss/L_t", L_t_valid.item(), cur_iter)
        writer.add_scalar("valid_loss/L_a", L_a_valid.item(), cur_iter)

        is_best = L_overall_valid < best_loss
        best_loss = min(L_overall_valid, best_loss)
        if is_best or (args.save_ckpt and epoch % 10 == 0):
            if not os.path.exists("./ckpt/"):
                os.makedirs("./ckpt/")
            save_checkpoint(epoch, model, optimizer, cur_iter, max_iter, L_overall.item(), is_best, args.ckpt_path)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


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
        # composite_dataset(args.raw_data_path, logger)
        gen_train_valid_names(args.valid_portion, logger)


if __name__ == "__main__":
    main()
