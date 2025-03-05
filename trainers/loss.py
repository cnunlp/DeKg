import torch
import torch.nn as nn

import numpy as np


class MIloss(nn.Module):
    def __init__(self, sigma=1):
        super(MIloss, self).__init__()
        self.sigma = sigma

    def pairwise_distance(self, x, y):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.norm(y, p=2, dim=0, keepdim=True)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist = torch.clamp(dist, min=0.0) 
        return dist
    def custom_cdist(self, X, p=2):

        assert X.dim() == 2, "Input X must be a 2D tensor"
        X_norm = (X**2).sum(dim=1, keepdim=True)  
        dist = X_norm + X_norm.t() - 2 * torch.mm(X, X.t()) 
        dist = torch.clamp(dist, min=0.0) 
        if p == 2:
            dist = dist.sqrt() 
        else:
            dist = dist.pow(p)  
        return dist
    def forward(self, W_t, W_clip):

        K_t = self.gaussian_kernel_matrix(W_t)
        K_clip = self.gaussian_kernel_matrix(W_clip)

        N = W_t.size(0)  
        H_t = self.entropy(K_t, N)
        H_clip = self.entropy(K_clip, N)
        
        H_joint = self.joint_entropy(K_t, K_clip, N)
        MI = H_t + H_clip - H_joint

        return MI

    def gaussian_kernel_matrix(self, X):

        # pairwise_distances = torch.cdist(X, X, p=2).half 
        pairwise_distances = torch.cdist(X.float(), X.float(), p=2)        
        pairwise_distances = pairwise_distances.to(torch.float16)

        kernel_matrix = torch.exp(-pairwise_distances ** 2 / (2 * self.sigma ** 2))  # 高斯核
        return kernel_matrix

    def entropy(self, K, N):
        
        P = K / K.sum(dim=1, keepdim=True)  
        H = -torch.sum(P * torch.log(P + 1e-8)) / N 
        return H
    
    def joint_entropy(self, K_t, K_clip, N):
        
        K_joint = K_t * K_clip  
        return self.entropy(K_joint, N)


class HSICLoss(nn.Module):
    def __init__(self):
        super(HSICLoss, self).__init__()

    def forward(self, W_t, W_clip):
        norm = torch.norm(W_t)
        W_t = W_t / norm
        norm = torch.norm(W_clip)
        W_clip = W_clip / norm

        n_cls = W_t.size(0)  
        device = W_t.device
        dtype = W_t.dtype
        # Computing the H-matrix
        # H = torch.eye(n_cls).to(W_t.device) - (1 / n_cls) * torch.ones(n_cls, n_cls).to(W_t.device)
        H = torch.eye(n_cls, device=device, dtype=dtype) - (1 / n_cls) * torch.ones(n_cls, n_cls, device=device, dtype=dtype)
        # Calculate K_t and K_clip
        K_t = torch.mm(W_t, W_t.t())
        K_clip = torch.mm(W_clip, W_clip.t())
        HSIC = torch.trace(torch.mm(H, torch.mm(K_t, torch.mm(H, K_clip))))

        return HSIC
