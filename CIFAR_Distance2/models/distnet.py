import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def gen_cluster_means(z_dim, n_classes):
    cluster_means = [torch.randn((n_classes + 1, z_dim))]

    cluster_means[0].requires_grad = True

    opt_cluster_means, opt_cluster_means_loss = None, float('inf')

    optimizer = optim.Adam(cluster_means, lr=1)

    for i in range(10000):
        c = cluster_means[0]
        c = c.div(c.norm(dim=1, keepdim=True))
        m = c @ c.T - 2 * torch.eye(n_classes + 1)
        loss = m.max(dim=1).values.mean()

        optimizer.zero_grad()
        loss.backward()
        if loss < opt_cluster_means_loss:
            opt_cluster_means_loss = loss
            opt_cluster_means = c.detach().clone()
        if loss < 0.1:
            break
        optimizer.step()
    return opt_cluster_means


class DistanceNet(nn.Module):
    def __init__(self, backbone, z_dim=256, n_classes=10):
        super(DistanceNet, self).__init__()
        self.z_dim = z_dim
        self.backbone = backbone
        self.cluster_means = nn.Parameter(gen_cluster_means(z_dim, n_classes)).requires_grad_(False)

    def forward(self, x):
        z = self.get_latent(x)
        o = self.get_in_distances(z)
        return o

    def get_latent(self, x):
        return self.backbone(x)

    def get_in_distances(self, z):
        cluster_means = self.cluster_means[:self.cluster_means.size(0)-1]

        n, d = z.size(0), z.size(1)
        m = cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = cluster_means.unsqueeze(0).expand(n, m, d)

        if d != cluster_means.size(1):
            raise Exception

        o = F.cosine_similarity(z_expanded, cluster_means_expanded, dim=2)
        o = 2 * o - o.pow(2)

        return o

    def get_out_distances(self, z):
        cluster_means = self.cluster_means[-1:]

        n, d = z.size(0), z.size(1)
        m = cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = cluster_means.unsqueeze(0).expand(n, m, d)

        if d != cluster_means.size(1):
            raise Exception

        o = F.cosine_similarity(z_expanded, cluster_means_expanded, dim=2)
        o = 2 * o - o.pow(2)

        return o

    def get_cluster_means(self):
        return self.cluster_means


class DistanceModule(nn.Module):
    def __init__(self, z_dim, n_classes):
        super(DistanceModule, self).__init__()
        self.cluster_means = nn.Parameter(gen_cluster_means(z_dim, n_classes)).requires_grad_(False)

    def forward(self, z):
        n, d = z.size(0), z.size(1)
        m = self.cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = self.cluster_means.unsqueeze(0).expand(n, m, d)

        if d != self.cluster_means.size(1):
            raise Exception

        o = F.cosine_similarity(z_expanded, cluster_means_expanded, dim=2)

        return o
