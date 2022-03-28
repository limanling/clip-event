# Code from https://raw.githubusercontent.com/pytorch/vision/master/references/detection/utils.py

import utils
import math
import sys
import tqdm
import logging
import torch
import time
import os

import model_clip
from utils import comm
from utils import WarmupCosineLR

def train_one_epoch(model, criterion, criterion_ot, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, writer_dict, distributed):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # for batch in data_loader:
    #     image_vec = batch.image_vec
    for batch in metric_logger.log_every(data_loader, print_freq, header): #tqdm.tqdm(data_loader): # #
        image_id = batch.image_id
        image_vec = batch.image_vec
        description = batch.description
        description_vec = batch.description_vec
        labels_per_image = batch.labels_per_image
        labels_per_text = batch.labels_per_text
        index_description_pos = batch.index_description_pos
        object_id = batch.object_id
        object_vec = batch.object_vec
        object_label = batch.object_label
        object_num = batch.object_num
        entitytxt_id = batch.entitytxt_id
        entitytxt_vec = batch.entitytxt_vec
        entitytxt_label = batch.entitytxt_label
        entitytxt_num = batch.entitytxt_num
        eventtxt_id = batch.eventtxt_id
        eventtxt_vec = batch.eventtxt_vec
        eventtxt_label = batch.eventtxt_label
        eventtxt_num = batch.eventtxt_num

        # print('parameters', list(model.parameters())[0]) #, list(model.parameters())[-1])

        # print('image_id', image_id, 'image_vec', image_vec.size(), 'description_vec', description_vec.size())
        logits_per_image, logits_per_text = model(image_vec, description_vec)
        # print(logits_per_image, logits_per_text)

        # loss: Contrastive loss
        constrastive_overbatch = model.module.constrastive_overbatch if distributed else model.constrastive_overbatch 
        loss_dict = criterion(logits_per_image, logits_per_text, labels_per_image, labels_per_text, index_pos=index_description_pos, constrastive_overbatch=constrastive_overbatch)
        # print('loss_dict', loss_dict)

        # loss: Alignment loss
        alignment = model.module.alignment if distributed else model.alignment
        if alignment:
            if distributed:
                image_features, text_features = model.module.sim_entity(object_vec, entitytxt_vec)
            else:
                image_features, text_features = model.sim_entity(object_vec, entitytxt_vec)
            loss_dict_ot = criterion_ot(text_features, image_features, entitytxt_num, object_num)
            loss_dict.update(loss_dict_ot)
            # print('loss_dict_ot', loss_dict_ot)
        
        losses = sum(loss for loss in loss_dict.values()) #/ len(loss_dict.values())
        # # print('losses', losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # print('loss_dict_reduced', loss_dict_reduced)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print('losses_reduced', losses_reduced)

        loss_value = losses_reduced.item()
        # loss_value = losses.item()

        if not math.isfinite(loss_value):
            logging.error("Loss is {}, stopping training".format(loss_value))
            logging.error(loss_dict_reduced)
            sys.exit(1)

        # print('lr', optimizer.param_groups[0]["lr"])
        # print('gradient', model.visual.conv1.weight.grad)
        # print('conv1 before', model.visual.conv1.weight)
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # print('gradient after', model.visual.conv1.weight.grad)
        # print('conv1 after', model.visual.conv1.weight)

        if lr_scheduler is not None:
            lr_scheduler.step()
            # print('lr', optimizer.param_groups[0]["lr"])
        
        # NOTE: add this follows clip code
        torch.cuda.synchronize()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', metric_logger.loss.global_avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return metric_logger


def build_criterion(args, train=False):
    # if args.cfg['loss'] == 'constrastive':
    criterion_constrastive = model_clip.CriterionContrastive(args.cfg['constrastive_loss'])
    # else:
    #     raise RuntimeError("Invalid criterion '{}'. ".format(args.cfg['loss']))

    if args.cfg['alignment']:
        criterion_ot = model_clip.CriterionAlignment()
    else:
        criterion_ot = None

    return criterion_constrastive, criterion_ot


def build_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    print('params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for name, param in model.named_parameters():
    #     print(name)

    if args.cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=args.cfg['lr'], 
            momentum=args.cfg['momentum'], 
            weight_decay=args.cfg['weight_decay']
            )
    elif args.cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.cfg['lr'],
            weight_decay=args.cfg['weight_decay'],
        )
    else:
        raise RuntimeError("Invalid optimizer '{}'. ".format(args.cfg['optimizer']))
    
    return optimizer


def build_lr_scheduler(args, optimizer, begin_epoch):
    if args.cfg['lr_scheduler'] == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.cfg['lr_steps'], gamma=args.cfg['lr_gamma'])
    elif args.cfg['lr_scheduler'] == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cfg['max_epoch'] - begin_epoch)
    elif args.cfg['lr_scheduler'] == 'warmup':
        # warmup_factor = 1. / 1000
        # warmup_iters = min(1000, data_loader_len - 1)
        # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        lr_scheduler = WarmupCosineLR(
            optimizer,
            args.cfg['max_epoch'],
            warmup_epochs=args.cfg['warmup_epoch'],
            last_epoch=begin_epoch - 1
        )
    elif args.cfg['lr_scheduler'] == 'none':
        lr_scheduler = None
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.cfg['lr_scheduler']))
    
    return lr_scheduler

def create_logger(args, tb_log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if args.distributed:
        log_file_name = '{}_{}_{}_rank{}.txt'.format(args.cfg['task'], phase, time_str, args.rank)
    else:
        log_file_name = '{}_{}_{}.txt'.format(args.cfg['task'], phase, time_str)
    log_file_path = os.path.join(tb_log_dir, log_file_name)
    if args.distributed:
        head = '%(asctime)-15s:[P:%(process)d]:' + 'Rank[{}/{}]'.format(args.rank, args.world_size) + ' %(message)s'
    else:
        head = '%(asctime)-15s:[P:%(process)d]:' + ' %(message)s'
    logging.basicConfig(
        filename=str(log_file_path), format=head
    )
    logger = logging.getLogger()
    if args.cfg['log_level'] == 'debug':
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)

def save_model_on_master(model, args, ckpt_dir, epoch, best_perf, optimizer):
    if not comm.is_main_process():
        return

    states = model.module.state_dict() if args.distributed else model.state_dict() # model.state_dict() #
    logging.info('=> saving checkpoint to {}'.format(ckpt_dir))
    save_dict = {
        'epoch': epoch, # 'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': args.cfg['task'],
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }
    try:
        torch.save(save_dict, os.path.join(ckpt_dir, '%s_%s.pth' % (args.cfg['task'], epoch)))#'checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')