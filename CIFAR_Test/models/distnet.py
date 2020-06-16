import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NaiveNet(nn.Module):
    def __init__(self, backbone, z_dim=1024, n_classes=10):
        super(PrioriNet, self).__init__()
        self.z_dim = z_dim
        self.backbone = backbone
        self.cluster_means = nn.Parameter(torch.zeros(n_classes, z_dim)).requires_grad_(True)

    def forward(self, x):
        z = self.get_latent(x)
        o = self.get_distances(z)
        return o

    def get_latent(self, x):
        return self.backbone(x)

    def get_distances(self, z):
        cluster_means = self.cluster_means

        n, d = z.size(0), z.size(1)
        m = cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = cluster_means.unsqueeze(0).expand(n, m, d)

        if d != cluster_means.size(1):
            raise Exception

        o = -torch.pow(z_expanded - cluster_means_expanded, 2).sum(2)

        return o

    def get_cluster_means(self):
        return self.cluster_means


def gen_euclidean_cluster_means(n_classes, z_dim):
    cluster_means = [torch.randn(n_classes, z_dim)]
    cluster_means[0].requires_grad = True

    opt_cluster_means, opt_cluster_means_loss = None, float('inf')

    optimizer = optim.Adam(cluster_means, lr=1)

    for i in range(10000):
        loss = (5 - torch.pdist(cluster_means[0])).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        if loss < opt_cluster_means_loss:
            opt_cluster_means_loss = loss
            opt_cluster_means = cluster_means[0].detach().clone()
        if loss < 1e-10:
            break
        optimizer.step()
    return opt_cluster_means.requires_grad_(False)


class PrioriNet(nn.Module):
    def __init__(self, backbone, z_dim=1024, n_classes=10):
        super(PrioriNet, self).__init__()
        self.z_dim = z_dim
        self.backbone = backbone
        self.cluster_means = nn.Parameter(gen_euclidean_cluster_means(n_classes, z_dim)).requires_grad_(False)

    def forward(self, x):
        z = self.get_latent(x)
        o = self.get_distances(z)
        return o

    def get_latent(self, x):
        return self.backbone(x)

    def get_distances(self, z):
        cluster_means = self.cluster_means

        n, d = z.size(0), z.size(1)
        m = cluster_means.size(0)

        z_expanded = z.unsqueeze(1).expand(n, m, d)
        cluster_means_expanded = cluster_means.unsqueeze(0).expand(n, m, d)

        if d != cluster_means.size(1):
            raise Exception

        o = -torch.pow(z_expanded - cluster_means_expanded, 2).sum(2)

        return o

    def get_cluster_means(self):
        return self.cluster_means


def gen_hyperspherical_cluster_means(z_dim, n_classes):
    cluster_means = [torch.randn((n_classes, z_dim))]
    cluster_means[0].requires_grad = True

    opt_cluster_means, opt_cluster_means_loss = None, float('inf')

    optimizer = optim.Adam(cluster_means, lr=1)

    for i in range(10000):
        c = cluster_means[0]
        m = c @ c.T - 2 * torch.eye(n_classes)
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


class HypersphericalNet(nn.Module):
    def __init__(self, backbone, z_dim=256, n_classes=10):
        super(HypersphericalNet, self).__init__()
        self.z_dim = z_dim
        self.backbone = backbone
        self.cluster_means = nn.Parameter(gen_hyperspherical_cluster_means(n_classes, z_dim)).requires_grad_(False)

    def forward(self, x):
        z = self.get_latent(x)
        o = self.get_distances(z)
        return o

    def get_latent(self, x):
        return self.backbone(x)

    def get_distances(self, z):
        cluster_means = self.cluster_means

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

