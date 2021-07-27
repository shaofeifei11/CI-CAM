import torch
from dataset.CUB_dataset import CUB_Dataset

from model.cub_vgg16 import CUB_VGG16
from model.cub_inceptionV3 import CUB_InceptionV3

from torch.utils.data import DataLoader
from utils import poly_lr_scheduler
from torch import nn
import numpy as np
import cv2
from utils import  net_load, net_save, old_create_attention
import config as cfg
from option import args_parser
from test2 import inference
from tensorboardX import SummaryWriter
import os
import time

if __name__ == '__main__':
    args = args_parser()

    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    gpus = args.gpu.replace("_", ",")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    print("gpus: ", gpus)

    if args.time == None:
        tim = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    else:
        tim = args.time
    logger = SummaryWriter(os.path.join("tensorboard", args.dir, tim))
    net_dir = os.path.join("save_model", args.dir, tim)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    if args.dataset == "cub":
        train_data = CUB_Dataset(input_size=args.input_size, train=True)
        if args.backbone == "vgg16":
            net = CUB_VGG16(args)
        elif args.backbone == "inceptionV3":
            net = CUB_InceptionV3(args)

    train_data_length = len(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

    cls_criterion = nn.CrossEntropyLoss().to(cfg.device)
    box_criterion = nn.SmoothL1Loss().to(cfg.device)

    begin_epoch = 0
    model_path = args.model_path
    if model_path != None:
        print("\n\n########################################################################")
        print("model_path: ", model_path)
        print("model_args: ", args.dir)
        net = net_load(net, model_path)
        begin_epoch = int(model_path.split("_")[-1].split(".")[0])

    print("run here! ", len(gpus))
    if len(gpus) > 1:
        print("DataParallel! ")
        net = torch.nn.DataParallel(net).to(cfg.device)
        net_module = net.module
    else:
        print("one machine! ")
        net = net.to(cfg.device)
        net_module = net

    if args.shared_classifier:
        optimizer = torch.optim.Adam([{'params': net_module.up_classifier.parameters()},
                                      {'params': net_module.regressor.parameters()},
                                      {'params': net_module.mask2attention.parameters()},
                                      {'params': net_module.backbone.parameters()}], lr=lr)
    else:
        optimizer = torch.optim.Adam([{'params': net_module.up_classifier.parameters()},
                                      {'params': net_module.down_classifier.parameters()},
                                      {'params': net_module.regressor.parameters()},
                                      {'params': net_module.mask2attention.parameters()},
                                      {'params': net_module.backbone.parameters()}], lr=lr)

    if args.attention:
        branch = 2
    else:
        branch = 1
    for e in range(begin_epoch, epoch):
        if args.decay:
            poly_lr_scheduler(optimizer, lr, e, lr_decay_iter=1, max_iter=epoch,
                              dataset=args.dataset,
                              backbone_rate=args.backbone_rate,
                              decay_rate=args.decay_rate, decay_epoch=args.decay_epoch)
        net.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_bbox_loss = 0
        for i, dat in enumerate(train_loader):
            images, labels = dat
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            labels = labels.long()
            optimizer.zero_grad()

            cam_up, out_up, pred_sort_up, pred_ids_up, cam_down, out_down, pred_sort_down, pred_ids_down = net(images)
            ############ up ###########
            cls_loss_up = cls_criterion(out_up, labels)
            loss = cls_loss_up
            epoch_loss += cls_loss_up.item()
            epoch_acc += float((pred_ids_up[:, 0].reshape(labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            ############ down ###########
            cls_loss_down = cls_criterion(out_down, labels)
            loss = loss + cls_loss_down
            epoch_loss += cls_loss_down.item()
            epoch_acc += float((pred_ids_down[:, 0].reshape(labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())

            if args.attention:
               net_module.update_mask(old_create_attention(labels=labels, pred_ids=pred_ids_up, cam=cam_up))

            print("epoch: ", e + 1, " | batch: ", i, "/", len(train_loader), " | loss: ", loss.item())

            loss.backward()
            optimizer.step()

        if args.dataset == "cub":
            if (e+1) % 2 == 0 or epoch - (e+1) < 5:
                net_save(net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 2 == 0:
                train_avg_epoch_acc = epoch_acc / (train_data_length*branch)
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l= inference(net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e+1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e+1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e+1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e+1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)
        else:
            net_save(net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 2 == 0:
                train_avg_epoch_acc = epoch_acc / (train_data_length*branch)
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l= inference(net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e + 1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e + 1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e + 1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e + 1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)

    logger.close()
