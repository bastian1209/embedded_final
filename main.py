import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.optim as optim
import argparse
from datetime import datetime
import os
import random
import warnings
import builtins

from configs.config import get_config, inspect_config, save_experiment_config, match_config_with_args
from data import get_loader, get_dataset
from methods import get_sscl_method
import time
from utils import AverageMeter, MetricLogger, accuracy, adjust_learning_rate
from utils import save_model, adjust_learning_rate, get_resume_info
from utils import Value


date = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--use_eqco', type=bool, default=False)  
parser.add_argument('--eqco_k', type=int, default=None)
parser.add_argument('--use_dcl', type=bool, default=False) 
parser.add_argument('--tau_plus', type=float, default=0.1)
parser.add_argument('--use_hcl', type=bool, default=False)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--save_date', type=str, default='{}_{}_{}_{}'.format(date.month, date.day, date.hour, date.minute))
parser.add_argument('--schedule', type=str, default='cos')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--world-size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--batch-size', type=float, default=256)
parser.add_argument('--tau', type=float, default=0.999) # 0.99 for ImageNet-100?
parser.add_argument('--num_view', type=int, default=2) 
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument('--data', type=str, default='ImageNet_100')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--dist-url', type=str, default='tcp://localhost:10001')
parser.add_argument('--method', type=str)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--use_symmetric_logit', type=bool, default=True) # for simclr
parser.add_argument('--epoch', type=int, default=200)


def main(config, args):
    config.defrost()
    if config.system.gpu is not None:
        warnings.warn('using a specific GPU => not available for torch.nn.parallel.DistributedDataParallel')
    
    if config.system.world_size == -1 and config.system.dist_url == "env://":
        config.system.world_size = int(os.environ['world_size'])
    
    config.system.distributed = config.system.world_size > 1 or config.system.multiprocessing_distributed
    
    num_gpus_per_node = torch.cuda.device_count()
    if config.system.multiprocessing_distributed:
        config.system.world_size = num_gpus_per_node * config.system.world_size
        config.freeze()
        mp.spawn(main_worker, nprocs=num_gpus_per_node, args=(num_gpus_per_node, config, args))
    else:
        config.freeze()
        main_worker(config.system.gpu, num_gpus_per_node, config)


def main_worker(gpu, num_gpus_per_node, config, args):
    config.defrost()
    config.system.gpu = gpu
    config.freeze()
    if config.system.multiprocessing_distributed and config.system.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    if config.system.gpu is not None:
        print("using {} gpu for training".format(config.system.gpu))
    
    # initialize distributed process group
    if config.system.distributed:
        config.defrost()
        if config.system.rank == -1 and config.system.dist_url == "env://":
            config.system.rank = int(os.environ["RANK"])
        if config.system.multiprocessing_distributed:
            config.system.rank = config.system.rank * num_gpus_per_node + int(config.system.gpu)
        
        dist.init_process_group(backend=config.system.dist_backend, 
                                init_method=config.system.dist_url, 
                                world_size=config.system.world_size, 
                                rank=config.system.rank)
        
    print('creating model : {}'.format(config.model.arch))
    model = get_sscl_method(config.method)(config)

    # distribute model
    if config.system.distributed:
        if (config.method == 'simclr'):
            nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder)
        # issue : need to implement SyncBN for Batchnorm1D for MLP => CVPODS can handle it but it requires PyTorch >= 1.3
        elif (config.method == 'byol'):
            pass
            # nn.SyncBatchNorm.convert_sync_batchnorm(model.online_network)
            # nn.SyncBatchNorm.convert_sync_batchnorm(model.target_network)
            
        if config.system.gpu is not None:
            torch.cuda.set_device(config.system.gpu)
            model.cuda(config.system.gpu)
            config.defrost()
            config.train.batch_size = int(config.train.batch_size / num_gpus_per_node)
            config.train.num_workers = int((config.system.num_workers + num_gpus_per_node - 1) / num_gpus_per_node)
            model = DDP(model, device_ids=[config.system.gpu])
            config.freeze()
        else:
            model.cuda()
            model = DDP(model)  
            
            raise NotImplementedError     
    else:
        raise NotImplementedError
    print(model)
    cudnn.benchmark = True
    
    # defining optimizer and scheduler
    optimizer = optim.__dict__[config.train.optim](model.parameters(), config.train.base_lr, momentum=config.train.momentum, weight_decay=config.train.wd)
    writer = SummaryWriter(log_dir=config.system.log_dir)
    
    total_step = Value()
    # resume
    if config.train.resume:
        print('resuming train...')
        if args.gpu is None:
            ckpt = torch.load(config.train.resume)
        else:
            map_location = "cuda:{}".format(args.gpu)
            ckpt = torch.load(config.train.resume, map_location=map_location)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('load params from {}'.format(config.train.resume))
        
        if 'total_step' in ckpt.keys():
            total_step = ckpt['total_step']
    
    # imagenet_100
    # defining dataset and data loader
    train_dataset = get_dataset(config, mode='train') 
    if config.system.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler=None
    train_loader = get_loader(config, train_dataset, train_sampler) 
    
    if config.method != 'byol':
        criterion = nn.CrossEntropyLoss().cuda(config.system.gpu)
    else:
        criterion = None
        
    # run train
    for epoch in range(config.train.start_epoch, config.train.epochs):
        if config.system.distributed:
            train_sampler.set_epoch(epoch) 
        adjust_learning_rate(optimizer, epoch, config)
        epoch_start = time.time()
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, total_step, config)
        epoch_end = time.time()
        print()
        print('EPOCH {} train time : {:.2f} min'.format(epoch, (epoch_end - epoch_start) / 60))
        
        if not config.system.multiprocessing_distributed or (config.system.multiprocessing_distributed 
                                                             and config.system.rank % num_gpus_per_node == 0):
            if (epoch + 1) % config.system.save_period == 0:
                filename = '{}/{}_{:04d}.pth.tar'.format(config.system.save_dir, config.model.arch, epoch + 1)
                save_model({"epoch" : epoch + 1, 
                            "optimizer" : optimizer.state_dict(), 
                            "model" : model.state_dict(), 
                            "arch" : config.model.arch, 
                            'total_step' : total_step }, filename)
                print('--------------------------- model saved at {} ---------------------------'.format(filename))
                print()
            else:
                print()
    filename = '{}/{}_final.pth.tar'.format(config.system.save_dir, config.model.arch)
    save_model({"epoch" : epoch + 1, 
                "optimizer" : optimizer.state_dict(), 
                "model" : model.state_dict(), 
                "arch" : config.model.arch, 
                'total_step' : total_step}, filename)
    print('############################ final model saved at {} ############################'.format(filename))


def train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, total_step, config):
    log_header = 'EPOCH {}'.format(epoch)
    losses = AverageMeter('Loss', fmt=':.4f')
    if config.method != 'byol':
        top1 = AverageMeter('Acc1', fmt=':4.2f')
        top5 = AverageMeter('Acc5', fmt=':4.2f')
    lr = AverageMeter('Lr', fmt=":.6f")
    
    metric_logger = MetricLogger(delimeter=" | ")
    metric_logger.add_meter(losses)
    if config.method != 'byol':
        metric_logger.add_meter(top1)
        metric_logger.add_meter(top5)
    metric_logger.add_meter(lr)
    # ce = nn.CrossEntropyLoss().cuda(config.system.gpu)
    # num_steps_per_epoch = int(len(train_loader.dataset) // config.train.batch_size)
    # global_step = num_steps_per_epoch * epoch
    for step, (images, _) in enumerate(metric_logger.log_every(train_loader, config.system.print_freq, log_header)):
        total_step.val += 1
        if config.system.gpu is not None:
            images[0] = images[0].cuda(config.system.gpu, non_blocking=True)
            images[1] = images[1].cuda(config.system.gpu, non_blocking=True)
        
        # [pos, neg]        
        # output = model(view_1=images[0], view_2=images[1])
        # loss, logits, targets = criterion(output)
        if config.method != 'byol':
            logits, targets, logits_original = model(view_1=images[0], view_2=images[1])
            loss = criterion(logits, targets)
            acc1, acc5 = accuracy(logits_original, targets, topk=(1, 5))
        else:
            loss_pre = model(view_1=images[0], view_2=images[1])
            loss = loss_pre.mean()
            
        lr_ = optimizer.param_groups[0]['lr']
        
        if config.method != 'byol':
            metric_logger.update(Loss=loss.detach().cpu().item(), 
                                Acc1=acc1.detach().cpu().item(), 
                                Acc5=acc5.detach().cpu().item(), 
                                Lr=lr_)
        else:
            metric_logger.update(Loss=loss.detach().cpu().item(), 
                                 Lr=lr_)
            
        writer.add_scalar('loss', loss.detach().cpu().item(), total_step.val)
        if config.method != 'byol':
            writer.add_scalar('top1', acc1.detach().cpu().item(), total_step.val)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.method == 'simclr':
        args.lr = 0.1
        args.temperature = 0.5
        args.weight_decay = 1e-6
    if args.method == 'byol':
        args.lr = 0.2
        args.weight_decay = 1.5e-6
        
    config = get_config(method=args.method)
    match_config_with_args(config, args)
    inspect_config(config)
    config.freeze()
    
    # save experiment config
    save_experiment_config(config)
    # run training
    main(config, args)