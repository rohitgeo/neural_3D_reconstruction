import torch
import torch.nn.functional as F
from torch import nn
from ray_utils import RayBundle
from functools import partial

# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)

        return torch.linalg.norm(
            sample_points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        diff = torch.abs(sample_points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# implicit_function:
#   type: nerf
#   n_harmonic_functions_xyz: 6
#   n_harmonic_functions_dir: 2
#   n_hidden_neurons_xyz: 128
#   n_hidden_neurons_dir: 64
#   density_noise_std: 0.0
#   n_layers_xyz: 6
#   append_xyz: [3]

# Implementation of NeRF
class NeRF(torch.nn.Module):
    def __init__(self, cfg):
        super(NeRF, self).__init__()

        self.emb_pts = HarmonicEmbedding(in_channels=3, 
                                         n_harmonic_functions=cfg.n_harmonic_functions_xyz)
        self.emb_dir = HarmonicEmbedding(in_channels=3, 
                                         n_harmonic_functions=cfg.n_harmonic_functions_dir)

        self.mlp = MLPWithInputSkips(n_layers=cfg.n_layers_xyz,
                                    input_dim=self.emb_pts.output_dim,
                                    output_dim=cfg.n_hidden_neurons_xyz,
                                    skip_dim=self.emb_pts.output_dim,
                                    hidden_dim=cfg.n_hidden_neurons_xyz,
                                    input_skips=cfg.append_xyz)

        self.density_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz, 1), nn.ReLU())

        self.color_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz + self.emb_dir.output_dim, cfg.n_hidden_neurons_dir),
                                         nn.ReLU(),
                                         nn.Linear(cfg.n_hidden_neurons_dir, 3),
                                         nn.Sigmoid())
    def forward(self, ray_bundle):
        
        pts = ray_bundle.sample_points
        dir = ray_bundle.directions

        point_embeddings = self.emb_pts(pts)

        out = self.mlp(point_embeddings, point_embeddings)
        
        density = self.density_head(out)

        dir_embeddings = self.emb_dir(dir).unsqueeze(1).repeat(1, pts.shape[1], 1)
        color = self.color_head(torch.cat([out, dir_embeddings], dim=-1))

        return {
            'density': density,
            'feature': color
        }


# Implementation of NeRF without view dependence
class NeRFWithoutViewDependence(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb_pts = HarmonicEmbedding(in_channels=3, 
                                         n_harmonic_functions=cfg.n_harmonic_functions_xyz)

        self.mlp = MLPWithInputSkips(n_layers=cfg.n_layers_xyz,
                                    input_dim=self.emb_pts.output_dim,
                                    output_dim=cfg.n_hidden_neurons_xyz,
                                    skip_dim=self.emb_pts.output_dim,
                                    hidden_dim=cfg.n_hidden_neurons_xyz,
                                    input_skips=cfg.append_xyz)

        self.density_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz, 1), nn.ReLU())

        self.color_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_dir),
                                         nn.ReLU(),
                                         nn.Linear(cfg.n_hidden_neurons_dir, 3),
                                         nn.Sigmoid())
    def forward(self, ray_bundle):
        
        pts = ray_bundle.sample_points
        point_embeddings = self.emb_pts(pts)
        out = self.mlp(point_embeddings, point_embeddings)
        
        density = self.density_head(out)
        color = self.color_head(out)

        out = {
            'density': density,
            'feature': color
        }
        # print(out)
        # exit(1)
        return out




class NeRFHiRes(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb_pts = HarmonicEmbedding(in_channels=3, 
                                         n_harmonic_functions=cfg.n_harmonic_functions_xyz)
        self.emb_dir = HarmonicEmbedding(in_channels=3, 
                                         n_harmonic_functions=cfg.n_harmonic_functions_dir)

        self.mlp = MLPWithInputSkipsMish(n_layers=cfg.n_layers_xyz,
                                    input_dim=self.emb_pts.output_dim,
                                    output_dim=cfg.n_hidden_neurons_xyz,
                                    skip_dim=self.emb_pts.output_dim,
                                    hidden_dim=cfg.n_hidden_neurons_xyz,
                                    input_skips=cfg.append_xyz)

        self.density_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz, 1), nn.ReLU(True))

        self.color_head = nn.Sequential(nn.Linear(cfg.n_hidden_neurons_xyz + self.emb_dir.output_dim, cfg.n_hidden_neurons_dir),
                                         nn.ReLU(True),
                                         nn.Linear(cfg.n_hidden_neurons_dir, 3),
                                         nn.Sigmoid())
    def forward(self, ray_bundle):
        
        pts = ray_bundle.sample_points
        dir = ray_bundle.directions

        point_embeddings = self.emb_pts(pts)

        out = self.mlp(point_embeddings, point_embeddings)
        
        density = self.density_head(out)

        dir_embeddings = self.emb_dir(dir).unsqueeze(1).repeat(1, pts.shape[1], 1)
        color = self.color_head(torch.cat([out, dir_embeddings], dim=-1))

        return {
            'density': density,
            'feature': color
        }

# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020 
# github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class MLPWithInputSkipsMish(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, Mish(), nn.BatchNorm1d(192)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y

volume_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeRF,
    'nerf_noviewdep': NeRFWithoutViewDependence,
    'nerf_hires': NeRFHiRes
}