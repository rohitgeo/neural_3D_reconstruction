import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_values = torch.linspace(start=self.min_depth, end=self.max_depth, 
                                  steps=self.n_pts_per_ray)
        z_values = z_values[None, :, None].repeat(len(ray_bundle.origins), 1, 3)

        directions = ray_bundle.directions[:, None, :].repeat(1, self.n_pts_per_ray, 1)

        # Sample points from z values
        sample_points = ray_bundle.origins[:, None, :].cuda() + directions * z_values.cuda()

        return ray_bundle._replace(sample_lengths=z_values.cuda() * torch.ones_like(sample_points[..., :1]).cuda(),
                            sample_points=sample_points)


sampler_dict = {
    'stratified': StratifiedRaysampler
}