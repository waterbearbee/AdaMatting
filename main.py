import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset.dataset import AdaMattingDataset
from dataset.pre_process import composite_dataset, gen_train_valid_names
from net.adamatting import AdaMatting
from loss import task_uncertainty_loss
from utility import get_args, get_logger, poly_lr_scheduler, save_checkpoint, AverageMeter, \
                    compute_mse, compute_sad


def train(model, optimizer, device, args, logger, multi_gpu):
    torch.manual_seed(7)
    writer = SummaryWriter()

    logger.info("Initializing data loaders")
    train_dataset = AdaMattingDataset(args.raw_data_path, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=16, pin_memory=True)
    valid_dataset = AdaMattingDataset(args.raw_data_path, 'valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                               num_workers=16, pin_memory=True)

    if args.resume:
        logger.info("Start training from saved ckpt")
        ckpt = torch.load(args.ckpt_path)
        model = ckpt["model"].module
        model = model.to(device)
        optimizer = ckpt["optimizer"]

        start_epoch = ckpt["epoch"] + 1
        max_iter = ckpt["max_iter"]
        cur_iter = ckpt["cur_iter"]
        init_lr = ckpt["init_lr"]
        best_loss = ckpt["best_loss"]
    else:
        logger.info("Start training from scratch")
        start_epoch = 0
        max_iter = 43100 * (1 - args.valid_portion) / args.batch_size * args.epochs
        cur_iter = 0
        init_lr = args.lr
        best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        torch.set_grad_enabled(True)
        model.train()
        for index, (img, gt) in enumerate(train_loader):
            cur_lr = poly_lr_scheduler(optimizer=optimizer, init_lr=init_lr, iter=cur_iter, max_iter=max_iter)

            img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
            gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
            gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

            optimizer.zero_grad()
            trimap_adaption, t_argmax, alpha_estimation = model(img)
            L_overall, L_t, L_a = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                        pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                        gt_alpha=gt_alpha, log_sigma_t_sqr=model.log_sigma_t_sqr, log_sigma_a_sqr=model.log_sigma_a_sqr)
            # if multi_gpu:
            #     L_overall, L_t, L_a = L_overall.mean(), L_t.mean(), L_a.mean()
            optimizer.zero_grad()
            L_overall.backward()
            optimizer.step()

            if cur_iter % 10 == 0:
                logger.info("Epoch: {:03d} | Iter: {:05d}/{} | Loss: {:.4e} | L_t: {:.4e} | L_a: {:.4e}"
                            .format(epoch, index, len(train_loader), L_overall.item(), L_t.item(), L_a.item()))
                writer.add_scalar("loss/L_overall", L_overall.item(), cur_iter)
                writer.add_scalar("loss/L_t", L_t.item(), cur_iter)
                writer.add_scalar("loss/L_a", L_a.item(), cur_iter)
                sigma_t = torch.exp(model.log_sigma_t_sqr / 2)
                sigma_a = torch.exp(model.log_sigma_a_sqr / 2)
                writer.add_scalar("sigma/sigma_t", sigma_t, cur_iter)
                writer.add_scalar("sigma/sigma_a", sigma_a, cur_iter)
                writer.add_scalar("lr", cur_lr, cur_iter)
            
            cur_iter += 1
        
        # Validation
        logger.info("Validating after the {}th epoch".format(epoch))
        avg_loss = AverageMeter()
        avg_l_t = AverageMeter()
        avg_l_a = AverageMeter()
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        model.eval()
        with tqdm(total=len(valid_loader)) as pbar:
            for index, (img, gt) in enumerate(valid_loader):
                img = img.type(torch.FloatTensor).to(device) # [bs, 4, 320, 320]
                gt_alpha = (gt[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device) # [bs, 1, 320, 320]
                gt_trimap = gt[:, 1, :, :].type(torch.LongTensor).to(device) # [bs, 320, 320]

                trimap_adaption, t_argmax, alpha_estimation = model(img)
                L_overall_valid, L_t_valid, L_a_valid = task_uncertainty_loss(pred_trimap=trimap_adaption, pred_trimap_argmax=t_argmax, 
                                                            pred_alpha=alpha_estimation, gt_trimap=gt_trimap, 
                                                            gt_alpha=gt_alpha, log_sigma_t_sqr=model.log_sigma_t_sqr, log_sigma_a_sqr=model.log_sigma_a_sqr)
                # if multi_gpu:
                #     L_overall, L_t, L_a = L_overall.mean(), L_t.mean(), L_a.mean()
                avg_loss.update(L_overall_valid.item())
                avg_l_t.update(L_t_valid.item())
                avg_l_a.update(L_a_valid.item())

                if index == 0:
                    trimap_adaption_res = torchvision.utils.make_grid(t_argmax.type(torch.FloatTensor) / 2, normalize=True, scale_each=True)
                    writer.add_image('valid_image/trimap_adaptation', trimap_adaption_res, cur_iter)
                    alpha_estimation_res = torchvision.utils.make_grid(alpha_estimation, normalize=True, scale_each=True)
                    writer.add_image('valid_image/alpha_estimation', alpha_estimation_res, cur_iter)
                
                pbar.update()

        logger.info("Average loss overall: {:.4e}".format(avg_loss.avg))
        logger.info("Average loss of trimap adaptation: {:.4e}".format(avg_l_t.avg))
        logger.info("Average loss of alpha estimation: {:.4e}".format(avg_l_a.avg))
        writer.add_scalar("valid_loss/L_overall", avg_loss.avg, cur_iter)
        writer.add_scalar("valid_loss/L_t", avg_l_t.avg, cur_iter)
        writer.add_scalar("valid_loss/L_a", avg_l_a.avg, cur_iter)

        is_best = avg_loss.avg < best_loss
        best_loss = min(avg_loss.avg, best_loss)
        if is_best or (args.save_ckpt and epoch % 10 == 0):
            if not os.path.exists("ckpts"):
                os.makedirs("ckpts")
            logger.info("Checkpoint saved")
            if (is_best):
                logger.info("Best checkpoint saved")
            save_checkpoint(epoch, model, optimizer, cur_iter, max_iter, init_lr, avg_loss.avg, is_best, args.ckpt_path)

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

    multi_gpu = False
    if args.mode != "prep":
        logger.info("Loading network")
        model = AdaMatting(in_channel=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        if args.cuda:
            device = torch.device("cuda:{}".format(device_ids[0]))
            if len(device_ids) > 1 and args.mode=="train":
                logger.info("Loading with multiple GPUs")
                model = torch.nn.DataParallel(model, device_ids=device_ids)
                multi_gpu = True
            model = model.cuda(device=device_ids[0])
        else:
            device = torch.device("cpu")

    if args.mode == "train":
        logger.info("Program runs in train mode")
        train(model=model, optimizer=optimizer, device=device, args=args, logger=logger, multi_gpu=multi_gpu)
    elif args.mode == "test":
        logger.info("Program runs in test mode")
        test()
    elif args.mode == "prep":
        logger.info("Program runs in prep mode")
        # composite_dataset(args.raw_data_path, logger)
        gen_train_valid_names(args.valid_portion, logger)


if __name__ == "__main__":
    main()
