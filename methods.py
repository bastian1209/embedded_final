import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from encoder import ResNet_SSL, MLPhead
from utils import schedule_byol_tau, concat_all_gather


def get_sscl_method(method_name="moco"):
    if method_name == 'moco':
        return MoCo
    elif method_name == 'simclr':
        return SimCLR
    elif method_name == 'byol':
        return BYOL


def EqCo(temperature, K, alpha):
    margin = temperature * np.log(alpha / K)
    
    return margin


class MoCo(nn.Module):
    def __init__(self, config):
        super(MoCo, self).__init__()
        self.config = config
        self.K = config.train.num_neg  
        self.tau = config.train.tau
        self.T = config.train.temperature
        
        # defining optimizing options : (1) Equivalent rule (2) debiasing (3) reweighting for hard negatives
        self.use_eqco = False
        self.margin = 0.
        if config.train.use_eqco:
            self.use_eqco = True
            self.alpha = self.K
            self.K = config.train.eqco_k
            self.margin = EqCo(self.T, self.K, self.alpha)
        
        self.use_dcl = False
        if config.train.use_dcl:
            self.use_dcl = True
            self.tau_plus = config.train.tau_plus
        
        self.use_hcl = False
        if config.train.use_hcl:
            self.use_hcl = True
            self.beta = config.train.beta
            
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'norm_layer' : config.model.normalize, 'is_cifar' : 'cifar' in config.dataset.name}
        self.query_encoder = ResNet_SSL(config.model.arch, config.model.head, 
                                      encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, method='moco')
        self.key_encoder = ResNet_SSL(config.model.arch, config.model.head, 
                                      encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, method='moco')
        self._init_encoders()
        self.register_buffer("neg_queue", torch.randn(self.ssl_feat_dim, self.K)) # [dim, K]
        self.neg_queue = F.normalize(self.neg_queue, dim=0)
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))
    
    def _init_encoders(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    @torch.no_grad()
    def _update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.tau * param_k.data + (1 - self.tau) * param_q.data
            
    @torch.no_grad()
    def _deque_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        pointer = int(self.queue_pointer) # self.queue_pointer.item()
        
        assert self.K % batch_size == 0
        self.neg_queue[:, pointer: pointer + batch_size] = keys.t()
        pointer = (pointer + batch_size) % self.K
        
        self.queue_pointer[0] = pointer
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        dist.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this]
        
    def forward(self, view_1, view_2):
        # already normalized
        q = self.query_encoder(view_1)
        
        with torch.no_grad():
            self._update_key_encoder()
            view_2, index_unshuffle = self._batch_shuffle_ddp(view_2)
            k = self.key_encoder(view_2) # already normalized
            k = self._batch_unshuffle_ddp(k, index_unshuffle)
        
        pos = torch.einsum('nd,nd->n', [q, k]).unsqueeze(-1)
        neg = torch.einsum('nd,dk->nk', [q, self.neg_queue.clone().detach()])
        
        pos_eqco = pos - self.margin # if self.use_eqco == False -> self.margin = 0
        if self.use_dcl:
            pos_exp = torch.exp(pos / self.T)
            neg_exp = torch.exp(neg / self.T)
            
            if self.use_hcl:
                importance = torch.exp(self.beta * neg / self.T)
                neg_exp = importance * neg_exp / importance.mean(dim=-1, keepdim=True)  
                          
            neg_exp = (-self.tau_plus * pos_exp + neg_exp) / (1 - self.tau_plus)
            neg_exp = torch.clamp(neg_exp, min=np.exp(-1 / self.T))
            
            pos_eqco_exp = torch.exp(pos_eqco / self.T)
            logits = torch.log(torch.cat([pos_eqco_exp, neg_exp], dim=1))
            
        else:
            logits = torch.cat([pos_eqco, neg], dim=1)
            logits = logits / self.T
            
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits_original = torch.cat([pos, neg], dim=1)
        
        self._deque_and_enqueue(k)
        
        return logits, labels, logits_original
            

class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        self.config = config
        self.alpha = 4096
        self.K = 256
        self.T = config.train.temperature
        self.use_symmetric_logit = config.train.use_symmetric_logit
        
        self.use_eqco = False
        self.margin = 0.
        if config.train.use_eqco:
            self.use_eqco = True
            self.K = config.train.eqco_k
            self.margin = EqCo(self.T, self.K, self.alpha)
        
        self.use_dcl = False
        if config.train.use_dcl:
            self.use_dcl = True
            self.tau_plus = config.train.tau_plus
            
        self.use_hcl = False
        if config.train.use_hcl:
            self.use_hcl = True
            self.beta = config.train.beta
         
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        self.encoder = ResNet_SSL(config.model.arch, config.model.head, 
                                  encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, method='simclr')
        
    def forward(self, view_1, view_2):
        batch_size = self.config.train.batch_size
        batch_size_this = view_1.shape[0]
        batch_size = view_1.shape[0]
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        """
        (1) symmetric_logit 
            K : 2 * batch_size - 2 per 2 queries
            N : 2 * batch_size 
        (2) normal
            K : batch_size - 1 per 1 query
            N : batch_size
        """
        assert batch_size % batch_size_this == 0
        
        if self.use_symmetric_logit:
            features = torch.cat([z_1, z_2], dim=0) # [2 * N, 128] # N = 256 per each gpu
            dot_similarities = torch.mm(features, features.t()) # [2 * N, 2 * ]
            
            pos_ij = torch.diag(dot_similarities, batch_size_this)
            pos_ji = torch.diag(dot_similarities, -batch_size_this)
            pos = torch.cat([pos_ij, pos_ji]).view(2 * batch_size_this, -1)
            
            diagonal = np.eye(2 * batch_size_this)
            pos_ij_eye = np.eye(2 * batch_size_this, k=batch_size_this)
            pos_ji_eye = np.eye(2 * batch_size_this, k=-batch_size_this)

            neg_mask = torch.from_numpy(1 - (diagonal + pos_ij_eye + pos_ji_eye)).cuda().type(torch.uint8)
            neg = dot_similarities[neg_mask].view(2 * batch_size_this, -1)

            if self.K < 256:
                assert self.use_eqco
                selection_mask = torch.stack([torch.cat([torch.ones(2 * self.K), torch.zeros(neg.shape[1] - 2 *  self.K)])[torch.randperm(neg.shape[1])] 
                                              for _ in range(2 * batch_size_this)], dim=0).cuda().type(torch.uint8)
                neg = neg[selection_mask].view(2 * batch_size_this, -1)
        else:
            dot_similarities = torch.mm(z_1, z_2.t()) # [N, N]
            pos = torch.diag(dot_similarities).unsqueeze(-1) # [N, 1]
            
            diagonal = torch.eye(batch_size)
            neg_mask = (1 - diagonal).cuda().type(torch.uint8)
            neg = dot_similarities[neg_mask].view(batch_size, -1) # [N, N - 1]

            if self.K < 256:
                one_zeros = torch.cat([torch.ones(self.K), torch.zeros(neg.shape[1] - self.K)])
                selection_mask = torch.stack([one_zeros[torch.randperm(neg.shape[1])] for _ in range(batch_size)], dim=0)
                selection_mask = selection_mask.cuda().type(torch.uint8)
                neg = neg[selection_mask].view(2 * batch_size, -1)
        
        pos_eqco = pos - self.margin
        if self.use_dcl:
            pos_exp = torch.exp(pos / self.T)
            neg_exp = torch.exp(neg / self.T)
            
            if self.use_hcl:
                importance = torch.exp(self.beta * neg / self.T)
                neg_exp = importance * neg_exp / importance.mean(dim=-1, keepdim=True)
            
            neg_exp = (-self.tau_plus * pos_exp + neg_exp) / (1 - self.tau_plus)
            neg_exp = torch.clamp(neg_exp, min=np.exp(-1 / self.T))
            
            pos_eqco_exp = torch.exp(pos_eqco / self.T)
            logits = torch.log(torch.cat([pos_eqco_exp, neg_exp], dim=1))
        else:
            logits = torch.cat([pos_eqco, neg], dim=1)
            logits = logits / self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits_original = torch.cat([pos, neg], dim=1)
        
        return logits, labels, logits_original


class BYOL(nn.Module):
    def __init__(self, config):
        super(BYOL, self).__init__()
        self.config = config
        
        self.tau = config.train.tau
        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'norm_layer' : config.model.normalize, 'is_cifar' : 'cifar' in config.dataset.name}
        self.online_network = ResNet_SSL(config.model.arch, config.model.head, encoder_params={'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}, ssl_feat_dim=self.ssl_feat_dim, method='byol')
        self.target_network = ResNet_SSL(config.model.arch, config.model.head, encoder_params={'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}, ssl_feat_dim=self.ssl_feat_dim, method='byol')
        self._init_encoders()
        
        hidden = self.online_network.proj_head.fc1.out_features
        self.predictor = MLPhead(2 * self.ssl_feat_dim, 2 * self.ssl_feat_dim, hidden=hidden, method='byol')
    
    def _init_encoders(self):
        for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
                
    @torch.no_grad()
    def _update_target_network(self):
        for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_online.data

    def forward(self, view_1, view_2):
        z_1_online = self.online_network(view_1)
        z_2_pred = self.predictor(z_1_online)
        z_2_pred = F.normalize(z_2_pred, dim=1)
        z_2_target = self.target_network(view_2)
        loss_1 = 2 - 2 * (z_2_pred * z_2_target).sum(dim=-1)
        
        z_2_online = self.online_network(view_2)
        z_1_pred = self.predictor(z_2_online)
        z_1_pred = F.normalize(z_1_pred, dim=1)
        z_1_target = self.target_network(view_1)
        loss_2 = 2 - 2 * (z_1_pred * z_1_target).sum(dim=-1)
        
        self._update_target_network()
        
        return loss_1 + loss_2


        

