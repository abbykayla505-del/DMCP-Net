import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Generalized_Homogeneity_Loss(nn.Module):

    def __init__(self, eps=1e-8, sigma=10.0, alpha=0.7):
        super().__init__()
        self.eps = eps
        self.sigma = sigma
        self.alpha = alpha

    @torch.no_grad()
    def _row_center_l2norm(self, X):
        mean = X.mean(dim=1, keepdim=True)
        Xc = X - mean
        l2 = torch.norm(Xc, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        U = Xc / l2
        return U

    def forward(self, X, P, Pos):
        """
        X: (N, d)
        P: (N, K)
        Pos: (N, 3)
        """

        eps = self.eps
        mass = P.sum(dim=0)  # (K,)

        # Homo Loss
        U = self._row_center_l2norm(X)  # (N, d)
        Feature_mean = (P.T @ U) / (mass.unsqueeze(1) + eps)
        Feature_intra_homo = (P * (U @ Feature_mean.T)).sum(dim=0) / (mass + eps)


        pos_mean = (P.T @ Pos) / (mass.unsqueeze(1) + eps)
        dist2 = (Pos ** 2).sum(dim=1, keepdim=True) + (pos_mean ** 2).sum(dim=1).unsqueeze(
            0) - 2 * Pos @ pos_mean.T  # (N, K)
        Pos_intra_homo = (P * torch.exp(-dist2 / (2.0 * self.sigma ** 2))).sum(dim=0) / (mass + eps)


        Weighted_intra_homo = (self.alpha * Pos_intra_homo + (1.0 - self.alpha) * Feature_intra_homo)

        H_min = 1e-4
        Weighted_intra_homo = Weighted_intra_homo.clamp(min=H_min, max=1.0)
        W = -torch.log(Weighted_intra_homo)
        unweighted_local_loss = torch.abs(1.0 - Weighted_intra_homo)
        weighted_local_loss = W * unweighted_local_loss
        weighted_intra_loss = weighted_local_loss.sum()
        Homo_loss = weighted_intra_loss

        # Balance loss
        N, d = X.shape
        probs = mass / (N + eps)
        empty_loss2 = (probs * torch.log(probs + eps)).sum()

        loss = Homo_loss+5*empty_loss2

        return loss,Feature_intra_homo.mean().item()

class Weighted_Active_Boundary_Loss(nn.Module):
    def __init__(self,
                 num_active_points=100,
                 max_clip_dist=10.0,
                 label_smoothing=0.0):

        super(Weighted_Active_Boundary_Loss, self).__init__()
        self.num_active_points = num_active_points
        self.max_clip_dist = max_clip_dist

        self.weight_func = lambda w: torch.clamp(w, max=self.max_clip_dist) / self.max_clip_dist

        self.criterion = nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=label_smoothing
        )

    def _kl_div_probs(self, p, q):

        q = q.detach()
        epsilon = 1e-8
        p = torch.clamp(p, min=epsilon)
        q = torch.clamp(q, min=epsilon)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

    def get_boundary_mask(self, probs, neighbor_indices):

        N, K = neighbor_indices.shape

        flat_neighbors = neighbor_indices.view(-1)
        valid_mask = (flat_neighbors >= 0)
        safe_indices = flat_neighbors.clone()
        safe_indices[~valid_mask] = 0
        safe_indices = safe_indices.long()

        neighbor_probs = probs[safe_indices].view(N, K, -1)
        center_probs = probs.unsqueeze(1)  # (N, 1, C)

        kl_values = self._kl_div_probs(center_probs, neighbor_probs)

        kl_values = kl_values * (neighbor_indices >= 0).float()

        max_kl_per_node, _ = kl_values.max(dim=1)

        k = min(self.num_active_points, N)
        _, topk_indices = torch.topk(max_kl_per_node, k=k)

        boundary_mask = torch.zeros(N, dtype=torch.bool, device=probs.device)
        boundary_mask[topk_indices] = True

        return boundary_mask, kl_values

    def forward(self, pred_probs, gt_dist_map, neighbor_indices, gradient):

        boundary_mask, kl_values = self.get_boundary_mask(pred_probs, neighbor_indices)

        if boundary_mask.sum() < 1:
            return pred_probs.sum() * 0.0

        N, K = neighbor_indices.shape
        safe_indices = neighbor_indices.clone()
        safe_indices[neighbor_indices == -1] = 0
        safe_indices = safe_indices.long()

        neighbor_dists = gt_dist_map[safe_indices]
        neighbor_dists[neighbor_indices == -1] = float('inf')

        direction_gt = torch.argmin(neighbor_dists, dim=1)

        active_kl_values = kl_values[boundary_mask]  # (M, K)
        active_direction_gt = direction_gt[boundary_mask]  # (M,)
        active_weights = gt_dist_map[boundary_mask]  # (M,)

        active_gradient = gradient[boundary_mask]  # (M,)

        active_neighbor_indices = neighbor_indices[boundary_mask]
        invalid_mask = (active_neighbor_indices == -1)
        active_kl_values[invalid_mask] = -1e9

        loss = self.criterion(active_kl_values, active_direction_gt.long().view(-1))  # (M,)

        w_dist = self.weight_func(active_weights)

        final_weight = w_dist * active_gradient

        loss = (loss * final_weight).mean()

        return loss

