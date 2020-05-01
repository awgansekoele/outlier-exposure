import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def gen_cluster_means(z_dim, n_classes):
    ood_dim = torch.cat((torch.ones(n_classes, 1) / z_dim, -torch.ones(n_classes, 1) / z_dim))
    cluster_means = [torch.randn((n_classes * 2, z_dim - 1))]

    cluster_means[0].requires_grad = True

    opt_cluster_means, opt_cluster_means_loss = None, float('inf')

    optimizer = optim.Adam(cluster_means, lr=1)

    for i in range(1000):
        c = cluster_means[0]
        c = c.div(c.norm(dim=1, keepdim=True)) * np.sqrt(1 - 1 / (z_dim ** 2))
        c = torch.cat((ood_dim, c), dim=1)
        m = c @ c.T - 2 * torch.eye(n_classes * 2)
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

        o = F.cosine_similarity(z_expanded, cluster_means_expanded, dim=2)

        return o

    def get_cluster_means(self):
        return self.cluster_means
