import torch
import torch.distributed as dist
import math
import numpy as np
from collections import deque, defaultdict, OrderedDict
from datetime import timedelta
import time
from encoder import ResNet_SL


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    return output


def schedule_byol_tau(tau_base, epoch, total_epoch):
    return 1 - (1 - tau_base) * (math.cos(math.pi * epoch / total_epoch) + 1) / 2


def adjust_learning_rate(optimizer, epoch, config, gamma=0.1):
    """Decay the learning rate based on schedule"""
    lr = config.train.base_lr
    if config.train.schedule == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config.train.epochs))
    else:  # stepwise lr schedule
        for milestone in config.train.milestones:
            lr *= gamma if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), only_count=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            if only_count:
                res.append(correct_k)
            else:
                res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def save_model(state_dict, filename):        
    torch.save(state_dict, filename) 
    

def load_sl_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    return ckpt['model']


def load_ssl_model(ckpt_path, method='moco'):
    ckpt = torch.load(ckpt_path)
    model_ssl_dict = ckpt['model']
    model_sl_dict = OrderedDict()
    
    if method == 'moco':
        prefix = 'module.query_encoder.'
        
    elif method == 'simclr':
        prefix = 'module.encoder.'
        
    elif method == 'byol':
        prefix = 'module.online_network.'
    
    for k, v in model_ssl_dict.items():
        if k.startswith(prefix):
            key = k.replace(prefix, '')
            model_sl_dict[key] = v
              
    return model_sl_dict


def get_sl_model(model_path, arch, dataset_name, num_classes, method, eval_only=False, use_bn=False):
    encoder_params = {'norm_layer' : None, 'is_cifar' : 'cifar' in dataset_name}
    model = ResNet_SL(arch_name=arch, encoder_params=encoder_params, num_classes=num_classes, use_bn=use_bn)
    
    if eval_only: 
        model_params = load_sl_model(model_path)
    else:
        model_params = load_ssl_model(model_path, method)
    msg = model.load_state_dict(model_params, strict=False)
    print(msg)
    
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    return model


def get_resume_info(ckpt_path):
    ckpt = torch.load(ckpt_path)
    optim_info = ckpt['optimizer']
    lr = list(map(lambda x : x['lr'], optim_info['param_groups']))
    assert len(set(lr)) == 1, "learning rare are not identical"
    epoch = ckpt['epoch']
    model_ckpt = ckpt['model']
    arch = ckpt['arch']
    
    return epoch, lr[0], model_ckpt, arch



class Value:
    def __init__(self):
        self.val = 0


class AverageMeter:
    def __init__(self, name, window_size=20, fmt=None):
        if fmt is None:
            fmt = '.4f'
        self.name = name
        self.deque = deque(maxlen=window_size)
        self.total = 0.
        self.count = 0
        self.fmt = "{avg" + fmt + "}" + "({global_avg" + fmt + "})"
    
    def update(self, val, n=1):
        self.deque.append(val)
        self.count += n
        self.total += val * n
        
    @property
    def median(self):
        data = torch.tensor(list(self.deque))
        return data.median().item()
    
    @property
    def avg(self):
        data = torch.tensor(list(self.deque), dtype=torch.float32)
        return data.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            max=self.max,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

 
class MetricLogger:
    def __init__(self, delimeter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimeter = delimeter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in  self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object does not have attribute '{}'".format(type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimeter.join(loss_str)
    
    def add_meter(self, meter):
        name = meter.name
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start = time.time()
        end = time.time()
        iter_time = AverageMeter('iter_time', fmt="{avg:.4f}")
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        log_msg =  self.delimeter.join([
            header, 
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}'
        ])
        
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self)))
            i += 1
            end = time.time()
            