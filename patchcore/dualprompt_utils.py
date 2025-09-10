import torch
import torch.nn.functional as F

@torch.no_grad()
def farthest_point_sampling(feats: torch.Tensor, m: int) -> torch.Tensor:
    """
    Greedy Farthest Point Sampling over (N, D) feature vectors; returns indices of size (m,)
    """
    assert feats.ndim == 2
    N = feats.size(0)
    m = min(m, N)
    device = feats.device

    # Normalize features to unit sphere (cosine distance equivalence)
    x = F.normalize(feats, dim=1)
    idx = torch.zeros(m, dtype=torch.long, device=device)
    # Start from a random point
    idx[0] = torch.randint(0, N, (1,), device=device)
    # Track minimum distance to chosen set for each point
    dist = torch.full((N,), float('inf'), device=device)
    last = x[idx[0]].unsqueeze(0)  # (1, D)

    for i in range(1, m):
        # Compute cosine similarity to the last chosen point
        sim = (x @ last.t()).squeeze(1)  # (N,)
        d = 1.0 - sim  # cosine distance
        dist = torch.minimum(dist, d)
        idx[i] = torch.argmax(dist)
        last = x[idx[i]].unsqueeze(0)
    return idx

def downsample_mask_to_feature(mask: torch.Tensor, size_hw):
    """
    Downsample a full-resolution mask to a feature map size.
    mask: (B, H_img, W_img) -> returns (B, H_feat, W_feat) using bilinear interpolation.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=size_hw, mode="bilinear", align_corners=False).squeeze(1)
