import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 255, 255, 255, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    TP, FP, TN, FN = 0, 0, 0, 0

    for samples, targets, names in metric_logger.log_every(data_loader, print_freq, header):
        b, L = samples.shape[0], samples.shape[1]
        H, W = samples.shape[-2], samples.shape[-1]
        samples = samples.reshape(b * L, 3, H, W).to(device, non_blocking=True)  # B, 3, H, W
        targets = targets.reshape(b * L).to(torch.float32).to(device, non_blocking=True)  # B

        patch_outputs = None
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs, patch_outputs = outputs[0].squeeze(), outputs[-1].squeeze()

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.loss_weight))
            loss = criterion(outputs, targets)
            metric_logger.update(cls_loss=loss.item())

            if patch_outputs is not None:
                ploss = criterion(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

                outputs = (outputs + patch_outputs) / 2
            
            preds = torch.sigmoid(outputs) > 0.5
            targets = targets.to(torch.bool)
            tp, fp, tn, fn = (targets & preds).sum(), (~targets & preds).sum(), (~targets & ~preds).sum(), (targets & ~preds).sum()
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    ACC = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged training stats:", metric_logger)
    print('Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1 score: {:.3f}'.format(ACC, precision, recall, F1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.loss_weight))
    TP, FP, TN, FN = 0, 0, 0, 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, name in metric_logger.log_every(data_loader, 100, header):
        b, L = images.shape[0], images.shape[1]
        H, W = images.shape[-2], images.shape[-1]
        images = images.reshape(b * L, 3, H, W).to(device, non_blocking=True)  # B, 3, H, W
        target = target.reshape(b * L).to(torch.float32).to(device, non_blocking=True)  # B
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            output, patch_output = output[0].squeeze(), output[-1].squeeze()

            loss = criterion(output, target)
            metric_logger.update(cls_loss=loss.item())
            if patch_output is not None:
                ploss = criterion(patch_output, target)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

                output = (output + patch_output) / 2

            preds = torch.sigmoid(output) > 0.5
            target = target.to(torch.bool)
            tp, fp, tn, fn = (target & preds).sum(), (~target & preds).sum(), (~target & ~preds).sum(), (target & ~preds).sum()
            TP, FP, TN, FN = TP + tp, FP + fp, TN + tn, FN + fn

        metric_logger.update(loss=loss.item())

    ACC = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)
    print('Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1 score: {:.3f}'.format(ACC, precision, recall, F1))
    print()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    for image_list, target, name in metric_logger.log_every(data_loader, 100, header):
        images1 = image_list[0].to(device, non_blocking=True)  # b, L, 3, H, W (b = B / L)
        b, L = images1.shape[0], images1.shape[1]
        H, W = images1.shape[-2], images1.shape[-1]
        images1 = images1.reshape(b * L, 3, H, W)  # B, 3, H, W
        target = target.reshape(b * L).to(torch.float32).to(device, non_blocking=True)  # B
        names = []
        for i in range(len(name)):
            volume_id = name[i].split('_')[1]
            slice_idx = int(name[i].split('_')[-1].split('-')[0]), int(name[i].split('_')[-1].split('-')[-1])
            for j in range(slice_idx[0], slice_idx[-1]):
                if args.task == 'LASC':
                    img_name = '{}_{}_{}'.format(args.task, volume_id, f'{j:02d}')
                elif args.task == 'BraTS' or args.task == 'KiTS':
                    img_name = '{}_{}_{}'.format(args.task, volume_id, f'{j:03d}')

                elif args.task == 'lasc':  # LASC
                    img_name = '{}_{}_{}'.format(args.task, volume_id, f'{j:02d}')
                elif args.task == 'BRATS' or args.task == 'kits23':  # BraTS or KiTS
                    img_name = '{}_{}_{}'.format(args.task, volume_id, f'{j:03d}')
                names.append(img_name)
        name = names
        batch_size = images1.shape[0]

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)  # b, L, 3, H, W (b = B / L)
                b, L = images.shape[0], images.shape[1]
                H, W = images.shape[-2], images.shape[-1]
                images = images.reshape(b * L, 3, H, W)  # B, 3, H, W
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                output, cls_attentions = model(images, return_att=True, n_layers=args.layer_index,
                                                            attention_type=args.attention_type)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)
                cls_attentions = cls_attentions.cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    img_name = name[b]
                    for cls_ind in range(args.nb_classes):
                        if target[b]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            if args.attention_dir:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        fname = os.path.join(args.cam_npy_dir, img_name + '_' + str(cls_score) + '.npy')
                        np.save(fname, cls_attention)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
