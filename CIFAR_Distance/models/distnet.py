import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def gen_cluster_means(size, C):
    cluster_means = [torch.randn(size)]
    cluster_means[0].requires_grad = True

    opt_cluster_means, opt_cluster_means_loss = None, float('inf')

    optimizer = optim.Adam(cluster_means, lr=1)

    for i in range(10000):
        cluster_means_l = torch.cat((cluster_means[0], torch.zeros(1, cluster_means[0].size(1))), dim=0)
        loss = (C - torch.pdist(cluster_means_l)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        if loss < opt_cluster_means_loss:
            opt_cluster_means_loss = loss
            opt_cluster_means = torch.tensor(cluster_means[0])
        if loss < 1e-10:
            break
        optimizer.step()
    return opt_cluster_means.requires_grad_(False)


class DistanceNet(nn.Module):
    def __init__(self, backbone, z_dim=256, n_classes=10):
        super(DistanceNet, self).__init__()
        self.z_dim = z_dim
        self.backbone = backbone
        self.cluster_means = nn.Parameter(gen_cluster_means((n_classes, z_dim), 10)).requires_grad_(False)

    def forward(self, x):
        z = self.get_latent(x)
        o = self.get_distances(z)
        return o

    def get_latent(self, x):
        return self.backbone(x)

    def get_distances(self, z):
        n, d = z.size(0), z.size(1)
        m = self.cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = self.cluster_means.unsqueeze(0).expand(n, m, d)

        if d != self.cluster_means.size(1):
            raise Exception

        o = -torch.pow(z_expanded - cluster_means_expanded, 2).sum(2)

        return o

    def get_cluster_means(self):
        return self.cluster_means
