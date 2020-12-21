import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from configs.config import get_config, inspect_config
from encoder import ResNet_SL
from utils import load_ssl_model, load_sl_model, save_model, adjust_learning_rate
from utils import  AverageMeter, MetricLogger, accuracy, get_resume_info, Value, get_sl_model
from data import get_dataset, get_loader
from augment import base_augment
from datetime import datetime
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


date = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--multi-gpu', type=bool, default=False)
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--save-period', type=int, default=10)
parser.add_argument('--date', type=str, default="{}_{}_{}_{}".format(date.month, date.day, date.hour, date.minute))
parser.add_argument('--use_bn', type=bool, default=False)
parser.add_argument('--use_ssl_aug', type=bool, default=False)
parser.add_argument('--eval-only', type=bool, default=False)
parser.add_argument('--schedule', type=str, default='step')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--data', type=str, default='cifar')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--trained-path', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--resume', type=str, default=None)
args = parser.parse_args()


def train_linear_one_epoch(train_loader, model, criterion, optimizer, config, device):
    log_header = 'EPOCH {}'.format(epoch + 1)
    losses = AverageMeter('Loss', fmt=':.4f')
    top1 = AverageMeter('Top1', fmt=':4.2f')
    top5 = AverageMeter('Top5', fmt=':4.2f')
    lr = AverageMeter('Lr', fmt=":.4f")
    
    metric_logger = MetricLogger(delimeter=" | ")
    metric_logger.add_meter(losses)
    metric_logger.add_meter(top1)
    metric_logger.add_meter(top5)
    metric_logger.add_meter(lr)
    
    for step, (img, target) in enumerate(metric_logger.log_every(train_loader, config.system.print_freq, log_header)):
        img = img.to(device)
        target = target.to(device)
        logit = model_sl(img)

        loss = criterion(logit, target)
        acc1, acc5 = accuracy(logit, target, topk=(1, 5))
        lr_ = optimizer.param_groups[0]['lr']
                
        metric_logger.update(Loss=loss.detach().cpu().item(),
                             Top1=acc1.detach().cpu().item(), 
                             Top5=acc5.detach().cpu().item(), 
                             Lr=lr_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def eval(model, val_loader):
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for img, target in val_loader:
            batch_size = img.shape[0]
            img = img.cuda()
            target = target.cuda()
            pred = model(img)
            
            correct = accuracy(pred, target, topk=(1,), only_count=True)[0]
            total_correct += correct
    
    dataset_size = len(val_loader.dataset)
    
    return 100. * (total_correct / dataset_size)
    
    
if __name__ == "__main__":
    if args.method == 'simclr':
        args.lr = 0.05
        
    config = get_config()
    config.defrost()
    config.train.base_lr = args.lr
    config.train.batch_size = args.batch_size
    config.system.save_period = args.save_period
    config.train.schedule = args.schedule
    config.train.milestones = [30, 40, 50] if config.dataset.name == 'ImageNet_100' else config.train.milestones
    config.method = args.method
    config.model.arch = args.arch
    config.dataset.name = args.data
    config.dataset.num_classes = args.num_classes
    if config.dataset.name.startswith('cifar'):
        config.dataset.img_size = [32, 32, 3]
    
    inspect_config(config)
    
    if not args.eval_only:
        experiment_name = '_'.join(args.model_path.split('/')[2].split('_')[:-4])
        print(experiment_name)
        
        config.system.save_dir = os.path.join('./linear', experiment_name)

        if not os.path.exists(config.system.save_dir):
            os.mkdir(config.system.save_dir)        

        train_epochs = args.epochs
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if args.use_ssl_aug:
            train_dataset = get_dataset(config, mode='linear_train')
        else:
            train_dataset = get_dataset(config, mode='linear')
        train_loader = get_loader(config, train_dataset)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        val_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True,
                                       transform=transforms.Compose([
                                            transforms.Resize(40),
                                            transforms.CenterCrop(32),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=True, drop_last=True)
        
        model_sl = get_sl_model(args.model_path, config.model.arch, config.dataset.name, config.dataset.num_classes, config.method)
        model_sl = model_sl.to(device)
                
        if args.resume:
            epoch, _, model_ckpt, _ = get_resume_info(args.resume)
            model_sl.load_state_dict(model_ckpt)
            config.train.start_epoch = epoch
        config.freeze()
        
        # with multi gpus
        args.multi_gpu = torch.cuda.device_count() > 1
        if args.multi_gpu:
            model_sl = nn.DataParallel(model_sl)

        update_params = [param for param in model_sl.parameters() if param.requires_grad == True]
        optimizer = optim.SGD(update_params, lr=config.train.base_lr, momentum=0.9)
        # if args.schedule == 'step':
        #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.train.milestones, gamma=0.2)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config.train.start_epoch, train_epochs):
            adjust_learning_rate(optimizer, epoch + 1, config, gamma=0.1)
            epoch_start = time.time()
            train_linear_one_epoch(train_loader, model_sl, criterion, optimizer, config, device)
            print()
            print('EPOCH {} train time : {:.2f} min'.format(epoch + 1, (time.time() - epoch_start) / 60))
            if (epoch + 1) % config.system.save_period == 0:
                filename = os.path.join(config.system.save_dir, "{}_{:04d}.pth.tar".format(config.model.arch, epoch + 1))
                model_state = model_sl.module.state_dict() if args.multi_gpu else model_sl.state_dict()     
                state_dict = {'model' : model_state, 
                              'optimizer' : optimizer.state_dict(), 
                              'epoch' : epoch + 1, 
                              'arch' : config.model.arch}
                save_model(state_dict, filename)
                print("####################### model saved at {} #######################".format(filename))
            val_acc = eval(model_sl, val_loader)
            print('\nEPOCH {} validation accuracy : {:.2f}\n'.format(epoch + 1, val_acc.cpu().item()))
            print()

        filename = os.path.join(config.system.save_dir, "{}_final.pth.tar".format(config.model.arch))
        save_model(state_dict, filename)
    
    else:
        config.freeze()
        print("start evaluating ...")
        print()
        
        val_dataset = get_dataset(config, mode='val')
        val_loader = get_loader(config, val_dataset)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_sl = get_sl_model(args.trained_path, config, config.method, device, True)
        acc = eval(model_sl, val_loader)
        experiment_name = args.trained_path.split('/')[2]
        print('{} - linear eval performance : {:.2f}'.format(experiment_name, acc.cpu().item()))
            
            
    
    