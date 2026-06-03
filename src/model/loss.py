import torch
from torch import nn
import torch.nn.functional as F


class PairwiseGaussianLoss(nn.Module):
    def __init__(self, n_classes: int, beta: float = 0.05):
        """Initialize pairwise Gaussian loss parameters."""
        super(PairwiseGaussianLoss, self).__init__()
        self.n_classes = n_classes
        self.beta = beta

    # reference from https://github.com/ccq1n/Pairwise_Gaussian_Loss/blob/master/loss_function.py#L12
    def forward(self, dist_mat: torch.Tensor, labels_raw: torch.Tensor):
        """Compute pairwise Gaussian loss from pair distances and labels."""
        device = dist_mat.device
        labels_raw = labels_raw.view([-1, 1])
        one_hot = torch.zeros(labels_raw.shape[0], self.n_classes).scatter_(1, labels_raw.data.cpu(), 1)
        dim_v = labels_raw.size(0)
        odd_list = [i for i in range(1, dim_v, 2)]
        even_list = [i for i in range(0, dim_v, 2)]
        labels_1 = one_hot[odd_list, :]
        labels_2 = one_hot[even_list, :]
        labels_ip = torch.max(labels_1*labels_2, dim=1, keepdim=True)[0].to(device)
        dist_mat_sq = self.beta * (torch.pow(dist_mat, 2))
        loss = dist_mat_sq + (labels_ip-1.0)*(torch.log(torch.exp(dist_mat_sq)) - 1.0)
        loss = torch.mean(loss)
        return loss

    # reference from https://github.com/ccq1n/Pairwise_Gaussian_Loss/blob/master/loss_function.py#L3
    @staticmethod
    def euclidean_dist_all(mat_ab):
        """Compute Euclidean distances between odd and even rows of an embedding matrix."""
        dim_v = mat_ab.size(0)
        odd_list = [i for i in range(1,dim_v,2)]
        even_list = [i for i in range(0,dim_v,2)]
        mat_a = mat_ab[odd_list, :]
        mat_b = mat_ab[even_list,:]
        dist_ab_eula = torch.sqrt(torch.sum(torch.pow((mat_a-mat_b), 2), dim=1, keepdim=True))
        return dist_ab_eula




"""
Reference: https://github.com/AI-Unicamp/Crab/blob/main/src/utils/losses.py
"""
def compute_cross_entropy(p, q):
    """Compute cross entropy between a target distribution and logits."""
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    """Shift logits by the per-row maximum for numerical stability."""
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss for single GPU/CPU training
    Based on: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        """Initialize multi-positive contrastive loss temperature."""
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embs, labels):
        """
        Args:
            embs: Embeddings tensor of shape [B, D]
            labels: Labels tensor of shape [B]

        Returns:
            loss: Scalar loss value
        """
        device = embs.device
        batch_size = embs.size(0)

        # Normalize embeddings
        feats = F.normalize(embs, dim=-1, p=2)

        # Compute similarity matrix
        logits = torch.matmul(feats, feats.T) / self.temperature

        # Create mask for positive pairs (same label)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)

        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Apply mask to logits (set self-similarity to very negative value)
        logits = logits - (1 - logits_mask) * 1e9

        # Stabilize logits
        logits = stablize_logits(logits)

        # Compute ground-truth distribution
        # Each positive pair gets equal probability
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # Compute loss
        loss = compute_cross_entropy(p, logits)

        return loss
