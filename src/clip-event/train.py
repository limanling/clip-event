import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.collect_env import get_pretty_env_info

from PIL import Image
import json
import os
import pprint
import logging
import tqdm
from tensorboardX import SummaryWriter
import time

import clip
from dataset_voa import VOADataset, VOADescriptionDataset
from engine import train_one_epoch, build_criterion, build_lr_scheduler, build_optimizer, create_logger, save_model_on_master
import utils
from utils import comm
from model_clip import build_model
from clip import _transform




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train clip event')

    parser.add_argument("--cfg", type=str, help="experiment configure file name", required=True)

    # # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', 
                        help='url used to set up distributed training')
    
    args = parser.parse_args()
    
    args.cfg = json.load(open(args.cfg))

    return args

def main():
    args = parse_args()

    args = utils.init_distributed_mode(args)
    print(args)

    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    torch.autograd.set_detect_anomaly(True) # change to training only?
    # setup_cudnn
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    print(args.cfg)

    # set up logger
    # logging.basicConfig(level=logging.DEBUG, filename=os.path.join(args.cfg['tb_log_dir'], args.cfg['task'], 'train-log.txt'))
    tb_log_dir = os.path.join(args.cfg['tb_log_dir'], args.cfg['task'], 'tensorboard')
    os.makedirs(tb_log_dir, exist_ok=True)
    log_dir = os.path.join(args.cfg['tb_log_dir'], args.cfg['task'], 'log')
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.cfg['ckpt_dir'], args.cfg['task'])
    os.makedirs(ckpt_dir, exist_ok=True)
    create_logger(args, log_dir, phase='train')

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info('config: {}'.format(str(args.cfg)))
        if args.distributed:
            logging.info("=> distributed using {} GPUs".format(args.world_size))

        output_config_path = os.path.join(tb_log_dir, 'config.json')
        logging.info("=> saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        if comm.is_main_process():
            json.dump(args.cfg, open(output_config_path, 'w'), indent=2)
    
    # tensorboard
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    
    # load model
    best_perf = 0.0
    best_model = True
    begin_epoch = args.cfg['begin_epoch']
    device = "cuda" if torch.cuda.is_available() else "cpu" #torch.device(args.gpu) #
    if args.cfg['jit']:
        # ## jit
        # model, preprocess = clip.load(args.cfg['begin_ckpt'], device=device, jit=False) 
        model = torch.jit.load(args.cfg['begin_ckpt'], map_location=device)
        state_dict = None
        model = build_model(state_dict or model.state_dict()).to(device)
        checkpoint_optimizer = None
    else:
        # ## non-jit, from checkpoint --> resume training (`begin_epoch` is accumulated)
        if os.path.exists(args.cfg['begin_ckpt']):
            logging.info(
                "=> loading checkpoint '{}'".format(args.cfg['begin_ckpt'])
            )
            # state_dict = torch.load(args.cfg['begin_ckpt'], map_location="cpu")
            # model = build_model(state_dict).to(device)  #state_dict or model.state_dict()).to(device)
            checkpoint_dict = torch.load(args.cfg['begin_ckpt'], map_location='cpu')
            best_perf = checkpoint_dict['perf']
            begin_epoch = checkpoint_dict['epoch' if args.cfg['is_train'] else 'step']  # checkpoint_dict['epoch' if in_epoch else 'step']
            checkpoint_optimizer = checkpoint_dict['optimizer']
            state_dict = checkpoint_dict['state_dict']
            model = build_model(state_dict).to(device)    # model.load_state_dict(state_dict)
        else:
            logging.error('=> error when loading checkpoint (cannot find checkpoint): {}'.format(args.cfg['begin_ckpt']))
            sys.exit(1)
    preprocess = _transform(model.visual.input_resolution)
    model.set_hyps(
        constrastive_overbatch=args.cfg['constrastive_overbatch'] if 'constrastive_overbatch' in args.cfg else False, 
        alignment=args.cfg['alignment'] if 'alignment' in args.cfg else False, 
        multiattention=args.cfg['multiattention'] if 'multiattention' in args.cfg else False
    )

    # load optimizer
    optimizer = build_optimizer(args, model)
    if checkpoint_optimizer is not None:
        # resume optimizer
        optimizer.load_state_dict(checkpoint_optimizer)
        logging.info(
            "=> {}: loaded checkpoint '{}' ({}: {})"
            .format(comm.head,
                    args.cfg['begin_ckpt'],
                    'epoch' if args.cfg['is_train'] else 'step',
                    begin_epoch)
        )
    lr_scheduler = build_lr_scheduler(args, optimizer, begin_epoch) #, len(data_loader_train))

    # build criterion
    criterion, criterion_ot = build_criterion(args)
    criterion.cuda()
    if criterion_ot: # the same as args.cfg['alignment']
        criterion_ot.cuda()
    criterion_val, criterion_ot_val = build_criterion(args, train=False)
    criterion_val.cuda()
    if criterion_ot_val:
        criterion_ot_val.cuda()

    # load data
    # dataset_train = VOADataset(
    #     image_caption_json_list=args.cfg['image_caption_json'], 
    #     image_dir_list=args.cfg['image_dir'], 
    #     preprocess=preprocess, 
    #     tokenize=clip.tokenize, 
    #     device=device
    # )
    dataset_train = VOADescriptionDataset(
        posneg_descriptions_json=args.cfg['posneg_descriptions_json'],
        image_caption_json_list=args.cfg['image_caption_json'], 
        image_dir_list=args.cfg['image_dir'], 
        # text ie
        load_ie=args.cfg['load_ie'], 
        ie_ontology_json=args.cfg['ie_ontology_json'], 
        input_entities=args.cfg['input_entities'],
        input_events=args.cfg['input_events'],
        ltf_dir=args.cfg['ltf_dir'],
        # object
        load_object=args.cfg['load_object'], 
        object_pickle=args.cfg['object_pickle'], 
        object_ontology_file=args.cfg['object_ontology_file'], 
        object_detection_threshold=args.cfg['object_detection_threshold'], 
        object_topk=args.cfg['object_topk'], 
        # situation recognition
        load_sr=args.cfg['load_sr'], 
        # model
        constrastive_overbatch=args.cfg['constrastive_overbatch'], 
        constrative_loss=args.cfg['constrastive_loss'], 
        preprocess=preprocess, 
        tokenize=clip.tokenize, 
        device=device
    )
    shuffle = True if args.cfg['is_train'] else False
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        # valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        shuffle = False
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        shuffle = False
        # valid_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # TODO: batch sampler, to make every batch using different event types
    # if args.aspect_ratio_group_factor >= 0:
    #     group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    # else:
    #     train_batch_sampler = torch.utils.data.BatchSampler(
    #         train_sampler, args.batch_size, drop_last=True)
 
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=args.cfg['batch_size'], 
        collate_fn=dataset_train.collate_fn,
        pin_memory=False,
        num_workers=0, #args.cfg['workers'],
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=True if args.cfg['is_train'] else False
    )

    # distributed model
    if args.distributed and args.cfg['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )

    # training
    for epoch in range(begin_epoch, args.cfg['max_epoch']):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # training for one epoch
        start = time.time()
        logging.info('=> {} train start'.format(head))
        train_one_epoch(model, criterion, criterion_ot, optimizer, lr_scheduler, data_loader_train, device, epoch, args.cfg['print_freq'], writer_dict, args.distributed)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate after every epoch
        

        # save
        save_model_on_master(model, args, ckpt_dir, epoch, best_perf, optimizer)



if __name__ == '__main__':
    main()
