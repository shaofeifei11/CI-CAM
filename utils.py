import torch
import numpy as np
import cv2
# from cupyx.scipy import ndimage
# import cupy

from scipy import ndimage
import os
import config as cfg
from torchvision.ops import RoIAlign, RoIPool
import pickle
import torch.nn.functional as F
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def _connected_component(im):
    mask = im > im.mean()
    label_im, nb_labels = ndimage.label(mask)


    object_slices = ndimage.find_objects(label_im)
    # Find the object with the largest area
    areas = [np.product([x.stop - x.start for x in slc]) for slc in object_slices]
    largest = object_slices[np.argmax(areas)]
    return [largest[0].start, largest[0].stop], [largest[1].start, largest[1].stop]


def connected_component(zero_one_map, h, w):
    """
    :param zero_one_map: [bz, h, w]
    :return:
    """
    bz = zero_one_map.shape[0]
    ids = torch.where(torch.eq(zero_one_map, torch.ones(zero_one_map.shape).to(cfg.device)),
                      torch.arange(h*w).unsqueeze(dim=0).expand(bz, h*w).float().to(cfg.device),
                      torch.mul(torch.ones(zero_one_map.shape).to(cfg.device), -w-1).to(cfg.device))
    shang = torch.div(ids, w).to(cfg.device).int()
    yu = torch.mul(torch.sub(torch.div(ids, w), shang), w).to(cfg.device)
    hh = torch.sort(shang, dim=1, descending=False)[0].float().to(cfg.device)
    ww = torch.sort(yu, dim=1, descending=False)[0].float().to(cfg.device)
    box_list = []
    for i in range(bz):
        box_list.append(torch.stack([ww[i, torch.gt(ww[i], -0.1).to(cfg.device)][0], hh[i, torch.gt(hh[i], -0.1).to(cfg.device)][0], ww[i, -1], hh[i, -1]], dim=0).to(cfg.device))
    return torch.stack(box_list, dim=0).to(cfg.device)

def _get_seg_bbox(cutting_image):
    """
    :param image_heatmap:  [height, width]
    :param seg_threshold:
    :return:
    """
    [slice_y, slice_x] = _connected_component(cutting_image)
    bounding_box = [slice_x[0], slice_y[0], slice_x[1], slice_y[1]]
    return torch.from_numpy(np.asarray(bounding_box)).to(cfg.device)

def get_target_from_bbox(seg_bbox, gt_bbox):
    """
    :param ex_rois:
    :param gt_rois:
    :return:
    """
    # gt_bbox = np.asarray(gt_bbox)
    ex_widths = torch.add(torch.sub(seg_bbox[:, 2], seg_bbox[:, 0]).to(cfg.device), 1.0).to(cfg.device)
    ex_heights = torch.add(torch.sub(seg_bbox[:, 3], seg_bbox[:, 1]).to(cfg.device), 1.0).to(cfg.device)
    ex_ctr_x = torch.add(seg_bbox[:, 0], torch.mul(ex_widths, 0.5).to(cfg.device)).to(cfg.device)
    ex_ctr_y = torch.add(seg_bbox[:, 1], torch.mul(ex_heights, 0.5).to(cfg.device)).to(cfg.device)

    gt_widths = torch.add(torch.sub(gt_bbox[:, 2], gt_bbox[:, 0]).to(cfg.device), 1.0).to(cfg.device)
    gt_heights = torch.add(torch.sub(gt_bbox[:, 3], gt_bbox[:, 1]).to(cfg.device), 1.0).to(cfg.device)
    gt_ctr_x = torch.add(gt_bbox[:, 0], torch.mul(gt_widths, 0.5).to(cfg.device)).to(cfg.device)
    gt_ctr_y = torch.add(gt_bbox[:, 1], torch.mul(gt_heights, 0.5).to(cfg.device)).to(cfg.device)

    targets_dx = torch.div(torch.sub(gt_ctr_x, ex_ctr_x).to(cfg.device), ex_widths).to(cfg.device)
    targets_dy = torch.div(torch.sub(gt_ctr_y, ex_ctr_y).to(cfg.device), ex_heights).to(cfg.device)
    targets_dw = torch.log(torch.div(gt_widths, ex_widths).to(cfg.device)).to(cfg.device)
    targets_dh = torch.log(torch.div(gt_heights, ex_heights).to(cfg.device)).to(cfg.device)

    targets = torch.transpose(torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), dim=0).to(cfg.device), 0, 1).to(cfg.device)
    return targets

def get_bbox_from_target(seg_bbox, target):
    """
    :param seg_bbox: segmentation the cam to get seg_bbox  .[4]
    :param target: the fc layer prediction  [4]
    :return:
    """

    # box = seg_bbox.astype(target.dtype, copy=False)
    box = seg_bbox
    width = torch.add(torch.sub(box[2], box[0]).to(cfg.device), 1.0).to(cfg.device)
    height = torch.add(torch.sub(box[3], box[1]).to(cfg.device), 1.0).to(cfg.device)
    ctr_x = torch.add(box[0], torch.mul(width, 0.5).to(cfg.device)).to(cfg.device)
    ctr_y = torch.add(box[1], torch.mul(height, 0.5).to(cfg.device)).to(cfg.device)

    dx = target[0]
    dy = target[1]
    dw = target[2]
    dh = target[3]

    pred_ctr_x = torch.add(torch.mul(dx, width).to(cfg.device), ctr_x).to(cfg.device)
    pred_ctr_y = torch.add(torch.mul(dy, height).to(cfg.device), ctr_y).to(cfg.device)
    pred_w = torch.mul(torch.exp(dw).to(cfg.device), width).to(cfg.device)
    pred_h = torch.mul(torch.exp(dh).to(cfg.device), height).to(cfg.device)

    pred_box = torch.zeros(target.shape).to(cfg.device)
    # x1
    pred_box[0] = torch.sub(pred_ctr_x, torch.mul(pred_w, 0.5).to(cfg.device)).to(cfg.device)
    # y1
    pred_box[1] = torch.sub(pred_ctr_y, torch.mul(pred_h, 0.5).to(cfg.device)).to(cfg.device)
    # x2
    pred_box[2] = torch.add(pred_ctr_x, torch.mul(pred_w, 0.5).to(cfg.device)).to(cfg.device)
    # y2
    pred_box[3] = torch.add(pred_ctr_y, torch.mul(pred_h, 0.5).to(cfg.device)).to(cfg.device)

    return pred_box


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=100, power=0.9, dataset="imagenet", backbone_rate=0.1, decay_rate=0.5, decay_epoch=2):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    lr = init_lr*(1 - iter/max_iter)**power
    if cfg.args.backbone == "inceptionV3" and iter >= 85:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if dataset == "imagenet":
        optimizer.param_groups[-3]['lr'] = optimizer.param_groups[-3]['lr'] * backbone_rate
        optimizer.param_groups[-2]['lr'] = optimizer.param_groups[-2]['lr'] * backbone_rate
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] * backbone_rate
        shang = iter // decay_epoch
        new_rate = 1
        for i in range(shang):
            new_rate = new_rate * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * new_rate
    else:
        optimizer.param_groups[-3]['lr'] = optimizer.param_groups[-3]['lr'] * backbone_rate
        optimizer.param_groups[-2]['lr'] = optimizer.param_groups[-2]['lr'] * backbone_rate
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] * backbone_rate

        shang = iter // decay_epoch
        new_rate = 1
        for i in range(shang):
            new_rate = new_rate * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * new_rate

def pred_bbox_resize_imagesize(boxA, image_size):
    """
    :param boxA:
    :param image_size:
    :return:
    """
    real_y = image_size[0].float()
    real_x = image_size[1].float()
    target_size = torch.tensor(cfg.crop_size).to(cfg.device)
    x_scale = torch.div(real_x, target_size).to(cfg.device)
    y_scale = torch.div(real_y, target_size).to(cfg.device)

    x = torch.mul(boxA[0], x_scale).to(cfg.device).float()
    y = torch.mul(boxA[1], y_scale).to(cfg.device).float()
    x_max = torch.mul(boxA[2], x_scale).to(cfg.device).float()
    y_max = torch.mul(boxA[3], y_scale).to(cfg.device).float()

    x = torch.where(x >= 0., x, torch.zeros([1]).to(cfg.device)[0]).to(cfg.device)
    y = torch.where(y >= 0., y, torch.zeros([1]).to(cfg.device)[0]).to(cfg.device)
    x_max = torch.where(x_max <= real_x, x_max, real_x).to(cfg.device)
    y_max = torch.where(y_max <= real_y, y_max, real_y).to(cfg.device)

    return torch.stack([x.int(), y.int(), x_max.int(), y_max.int()], dim=0).to(cfg.device)


def IoU(boxA, boxB):
    xA = boxA[0] if boxA[0] > boxB[0] else boxB[0]
    yA = boxA[1] if boxA[1] > boxB[1] else boxB[1]
    xB = boxA[2] if boxA[2] < boxB[2] else boxB[2]
    yB = boxA[3] if boxA[3] < boxB[3] else boxB[3]

    interArea = (0 if 0 > (xB - xA + 1) else (xB - xA + 1)) * (0 if 0 > (yB - yA + 1) else (yB - yA + 1))

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / (boxAArea + boxBArea - interArea).float()

    return iou.detach().cpu().numpy()

def localization_acc(boxA, boxB_list, class_match):
    if class_match == False:
        return 0.0
    max_iou = 0
    for i in range(len(boxB_list)):
        iou = IoU(boxA, boxB_list[i])
        max_iou = max(max_iou, iou)
    if max_iou >= 0.5:
        return 1.0
    else:
        return 0.0

def create_h_x(pred):
    pred_mean = torch.mean(pred)
    confidence = torch.sub(pred, pred_mean)
    conf_max = torch.max(torch.abs(confidence))
    return confidence / conf_max


def torch_cam_combination(cam, pred_sort, pred_ids, function, mean_num):
    """
    :param cam:  [bz, num_classes, height, width]
    :param pred_sort: score [bz, num_classes]
    :param pred_ids:  [bz, num_classes]
    :return:
    """
    bz, nc, h, w = cam.shape
    cam_bz_nc_index = get_pre_two_source_inds(cam.shape)
    tmp = cam[cam_bz_nc_index, pred_ids]

    # h_x = create_h_x(pred_sort[image_inds])
    if function == "linear":
        tmp = tmp.permute(1, 0, 2, 3)
        tmp = tmp.reshape((nc, bz * h * w))
        h_x = torch.linspace(1, -1, nc).to(cfg.device)
        activation_maps = torch.matmul(h_x, tmp).to(cfg.device)
    elif function == "quadratic":
        tmp = tmp.permute(1, 0, 2, 3)
        tmp = tmp.reshape((nc, bz * h * w))
        h_x = torch.linspace(0, -1, nc).to(cfg.device)  # quadratic
        h_x = torch.mul(h_x, h_x).to(cfg.device)
        h_x = - h_x
        activation_maps = torch.matmul(h_x, tmp).to(cfg.device)
    elif function == "sum_1":
        tmp = tmp.permute(1, 0, 2, 3)
        tmp = tmp.reshape((nc, bz * h * w))
        select_cam = tmp[-mean_num:]
        h_x = torch.linspace(0, 2, mean_num).unsqueeze(dim=1).expand((mean_num, bz * h * w)).to(cfg.device)
        select_cam = select_cam * h_x
        activation_maps = tmp[0]
        activation_maps = torch.sub(activation_maps, torch.div(torch.sum(select_cam, 0), mean_num)).to(cfg.device)
    elif function == "sum_2":
        tmp = tmp.permute(1, 0, 2, 3)
        tmp = tmp.reshape((nc, bz * h * w))
        select_cam = tmp[-mean_num:]
        h_x = torch.linspace(0, 4, mean_num).unsqueeze(dim=1).expand((mean_num, bz * h * w)).to(cfg.device)
        select_cam = select_cam * h_x
        activation_maps = tmp[0]
        activation_maps = torch.sub(activation_maps, torch.div(torch.sum(select_cam, 0), mean_num)).to(cfg.device)
    else:
        sub_tmp = torch.mean(tmp[:, -mean_num:, :, :], dim=1).to(cfg.device)
        activation_maps = torch.sub(tmp[:, 0, :, :], sub_tmp).to(cfg.device)

    activation_maps = activation_maps.reshape(bz, h * w)
    activation_maps = torch.sub(activation_maps, torch.min(activation_maps, dim=1, keepdim=True)[0]).to(cfg.device)
    return activation_maps

def old_create_attention(labels, pred_ids, cam):

    bz, nc, h, w = cam.shape
    attention4 = torch.zeros(size=[bz, nc, cfg.attention_size, cfg.attention_size],
                             requires_grad=False).to(cfg.device)
    ind = torch.eq(pred_ids[:, 0], labels)
    if len(ind) > 0:
        source_inds = get_pre_two_source_inds(shap=attention4.shape)
        attention4 = attention4[source_inds, pred_ids]  # reverse
        tmp = cam[source_inds, pred_ids]  # reverse
        attention4[ind, 0] = tmp[ind, 0]
        attention4 = attention4[source_inds, torch.argsort(pred_ids, dim=1).to(cfg.device)]  # reverse back
    else:
        print("create_attention ind is None")
    attention = torch.sum(attention4, dim=0).to(cfg.device)
    return attention

def compute_gt_rois(pred_sort, pred_ids, cam, upsample, seg_thr, labels, combination=False, function="quadratic", mean_num=1):
    """
    :param pred:  [batch_size, classes]
    :param cam:  [batch_size, channel, height, width]
    :return:
    """
    gt_rois = []
    bz, nc, h, w = cam.shape
    if combination:
        activation_maps = torch_cam_combination(cam, pred_sort, pred_ids, function, mean_num)
        activation_maps = activation_maps.reshape(bz, h, w)
        activation_maps = upsample(activation_maps.unsqueeze(dim=1).to(cfg.device)).to(cfg.device).squeeze(
                dim=1).to(cfg.device)
        activation_maps = activation_maps.reshape([bz, cfg.crop_size * cfg.crop_size])
        bounding_box_thr = torch.mul(torch.max(activation_maps, dim=1, keepdim=True)[0], seg_thr).to(cfg.device)
        cutting_images = torch.where(torch.gt(activation_maps, bounding_box_thr).to(cfg.device), activation_maps,
                                         torch.zeros(activation_maps.shape).to(cfg.device))
        cutting_images = cutting_images.reshape([bz, cfg.crop_size, cfg.crop_size]).detach().cpu().numpy()
        for image_inds in range(bz):
            gt_rois.append(_get_seg_bbox(cutting_images[image_inds]))
    else:
        activation_maps = cam.reshape(bz, nc, h * w)
        activation_maps = torch.sub(activation_maps, torch.min(activation_maps, dim=2, keepdim=True)[0]).to(cfg.device)
        activation_maps = activation_maps.reshape(bz, nc, h, w)

        activation_maps = upsample(activation_maps).to(cfg.device)
        activation_maps = activation_maps.reshape([bz, nc, cfg.crop_size * cfg.crop_size])
        bounding_box_thr = torch.mul(torch.max(activation_maps, dim=2, keepdim=True)[0], seg_thr).to(cfg.device)
        cutting_images = torch.where(torch.gt(activation_maps, bounding_box_thr).to(cfg.device), activation_maps,
                                         torch.zeros(activation_maps.shape).to(cfg.device))
        cutting_images = cutting_images.reshape([bz, nc, cfg.crop_size, cfg.crop_size]).detach().cpu().numpy()
        for image_inds in range(bz):
            gt_rois.append(_get_seg_bbox(cutting_images[image_inds, labels[image_inds]]))
    gt_rois = torch.stack(gt_rois, dim=0).to(cfg.device)
    return gt_rois  # [batch, 4]

def create_rois(pred_sort, pred_ids, cam, upsample, seg_thr, topk=1, combination=False, function="quadratic", mean_num=1):
    """
    :param pred: [batchz_size, num_classes]
    :param cam: [batch_size, channel, height 14, width 14]
    :return:
    """
    bz, nc, h, w = cam.shape
    if combination:
        activation_maps = torch_cam_combination(cam, pred_sort, pred_ids, function, mean_num)

        activation_maps = activation_maps.reshape(bz, h, w)

        activation_maps = upsample(activation_maps.unsqueeze(dim=1).to(cfg.device)).to(cfg.device).squeeze(dim=1).to(cfg.device)

        bbox_list = []
        activation_maps = activation_maps.reshape([bz, cfg.crop_size*cfg.crop_size])
        bounding_box_thr = torch.mul(torch.max(activation_maps, dim=1, keepdim=True)[0], seg_thr).to(cfg.device)
        cutting_images = torch.where(torch.gt(activation_maps, bounding_box_thr).to(cfg.device), activation_maps,
                                     torch.zeros(activation_maps.shape).to(cfg.device))
        cutting_images = cutting_images.reshape([bz, cfg.crop_size, cfg.crop_size]).detach().cpu().numpy()
        for image_inds in range(bz):
            bbox = _get_seg_bbox(cutting_images[image_inds])
            bbox_list.append(bbox.unsqueeze(dim=0).expand((topk, 4)).to(cfg.device))
        bbox_list = torch.stack(bbox_list, dim=0).to(cfg.device)
        return bbox_list  # [batch, topk, 4]

    else:
        cam_bz_nc_index = get_pre_two_source_inds(cam.shape)
        activation_maps = cam[cam_bz_nc_index, pred_ids].reshape(bz, nc, h * w)
        activation_maps = torch.sub(activation_maps, torch.min(activation_maps, dim=2, keepdim=True)[0]).to(cfg.device)
        activation_maps = activation_maps.reshape(bz, nc, h, w)

        activation_maps = upsample(activation_maps).to(cfg.device)
        bbox_list = []
        activation_maps = activation_maps.reshape([bz, nc, cfg.crop_size * cfg.crop_size])
        bounding_box_thr = torch.mul(torch.max(activation_maps, dim=2, keepdim=True)[0], seg_thr).to(cfg.device)
        cutting_images = torch.where(torch.gt(activation_maps, bounding_box_thr).to(cfg.device), activation_maps,
                                     torch.zeros(activation_maps.shape).to(cfg.device))
        cutting_images = cutting_images.reshape([bz, nc, cfg.crop_size, cfg.crop_size]).detach().cpu().numpy()
        for image_inds in range(bz):
            li = []
            for j in range(topk):
                li.append(_get_seg_bbox(cutting_images[image_inds, j]))
            bbox_list.append(torch.stack(li, dim=0).to(cfg.device))
        bbox_list = torch.stack(bbox_list, dim=0).to(cfg.device)
        return bbox_list  # [batch, topk, 4]


def path_from_net_to_mask(net_path):
    if net_path.find("net_train_") >= 0:
        return net_path.replace("net_train_", "mask_train_")
    else:
        return "non"

def net_save(net, net_path):
    torch.save(net.state_dict(), net_path)
    with open(path_from_net_to_mask(net_path), "wb") as f:
        pickle.dump(net.mask, f)

def net_load(net, net_path):
    weight = torch.load(net_path)
    net.load_state_dict(weight)
    net = net.to(cfg.device)
    mask_path = path_from_net_to_mask(net_path)
    if os.path.exists(mask_path):
        with open(path_from_net_to_mask(net_path), "rb") as f:
            net.mask = pickle.load(f)
    else:
        print("\n########## Attention: mask file is not exist.\n")
    return net

def get_pre_two_source_inds(shap):
    bz, nc = shap[0:2]
    return torch.arange(bz).unsqueeze(dim=1).expand((bz, nc)).to(cfg.device)
