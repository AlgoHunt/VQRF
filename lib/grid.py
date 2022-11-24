import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.cpp_extension import load
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# render_utils_cuda = load(
#         name='render_utils_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
#         verbose=True)

# total_variation_cuda = load(
#         name='total_variation_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
#         verbose=True)

import dvgo_cu
from dvgo_cu import render_utils_cuda, total_variation_cuda


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    else:
        raise NotImplementedError

from functools import lru_cache

@lru_cache(100)
def get_scaling_factor(width, height, num_d, device):
    return torch.FloatTensor([width-1.0, height-1.0, num_d-1.0]).to(device).view(1, 1, 1, 3)

class TrilinearIntepolation(nn.Module):
    """TrilinearIntepolation in PyTorch."""

    def __init__(self):
        super(TrilinearIntepolation, self).__init__()

    def sample_at_integer_locs(self, input_feats, index_tensor):
        assert input_feats.ndimension()==5, 'input_feats should be of shape [B,F,D,H,W]'
        assert index_tensor.ndimension()==4, 'index_tensor should be of shape [B,H,W,3]'
        # first sample pixel locations using nearest neighbour interpolation
        batch_size, num_chans, num_d, height, width = input_feats.shape
        grid_height, grid_width = index_tensor.shape[1],index_tensor.shape[2]
        xy_grid = index_tensor[..., 0:2]
        xy_grid[..., 0] = xy_grid[..., 0] - ((width-1.0)/2.0)
        xy_grid[..., 0] = xy_grid[..., 0] / ((width-1.0)/2.0)
        xy_grid[..., 1] = xy_grid[..., 1] - ((height-1.0)/2.0)
        xy_grid[..., 1] = xy_grid[..., 1] / ((height-1.0)/2.0)

        xy_grid = torch.clamp(xy_grid, min=-1.0, max=1.0)
        sampled_in_2d = F.grid_sample(input=input_feats.view(batch_size, num_chans*num_d, height, width),
                                        grid=xy_grid, mode='nearest').view(batch_size, num_chans, num_d, grid_height, grid_width)
        z_grid = index_tensor[..., 2].view(batch_size, 1, 1, grid_height, grid_width)
        z_grid = z_grid.long().clamp(min=0, max=num_d-1)
        z_grid = z_grid.expand(batch_size,num_chans, 1, grid_height, grid_width)
        sampled_in_3d = sampled_in_2d.gather(2, z_grid).squeeze(2)
        return sampled_in_3d
    
    def forward(self, input_feats, sampling_grid, vq):
        assert input_feats.ndimension()==5, 'input_feats should be of shape [B,F,D,H,W]'
        assert sampling_grid.ndimension()==4, 'sampling_grid should be of shape [B,H,W,3]'
        batch_size, num_chans, num_d, height, width = input_feats.shape
        grid_height, grid_width = sampling_grid.shape[1],sampling_grid.shape[2]
        # make sure sampling grid lies between -1, 1
        sampling_grid = torch.clamp(sampling_grid, min=-1.0, max=1.0)
        # map to 0,1
        sampling_grid = (sampling_grid+1)/2.0
        # Scale grid to floating point pixel locations
        # import ipdb;ipdb.set_trace()
        scaling_factor = get_scaling_factor(width, height, num_d, input_feats.device)
        sampling_grid = scaling_factor*sampling_grid
        # Now sampling grid is between [0, w-1; 0,h-1; 0,d-1]
        x, y, z = torch.split(sampling_grid, split_size_or_sections=1, dim=3)
        x_0, y_0, z_0 = torch.split(sampling_grid.floor(), split_size_or_sections=1, dim=3)
        x_1, y_1, z_1 = x_0+1.0, y_0+1.0, z_0+1.0
        u, v, w = x-x_0, y-y_0, z-z_0
        u, v, w = map(lambda x:x.view(batch_size, 1, grid_height, grid_width).expand(
                                    batch_size, num_chans, grid_height, grid_width),  [u, v, w])
        c_000 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_0], dim=3))
        c_001 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_0, z_1], dim=3))
        c_010 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_0], dim=3))
        c_011 = self.sample_at_integer_locs(input_feats, torch.cat([x_0, y_1, z_1], dim=3))
        c_100 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_0], dim=3))
        c_101 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_0, z_1], dim=3))
        c_110 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_0], dim=3))
        c_111 = self.sample_at_integer_locs(input_feats, torch.cat([x_1, y_1, z_1], dim=3))
        # import ipdb;ipdb.set_trace()
        c_000 = vq(c_000.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_001 = vq(c_001.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_010 = vq(c_010.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_011 = vq(c_011.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_100 = vq(c_100.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_101 = vq(c_101.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_110 = vq(c_110.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)
        c_111 = vq(c_111.squeeze(2).permute(0,2,1))[0].permute(0,2,1).unsqueeze(2)

        c_xyz = (1.0-u)*(1.0-v)*(1.0-w)*c_000 + \
                (1.0-u)*(1.0-v)*(w)*c_001 + \
                (1.0-u)*(v)*(1.0-w)*c_010 + \
                (1.0-u)*(v)*(w)*c_011 + \
                (u)*(1.0-v)*(1.0-w)*c_100 + \
                (u)*(1.0-v)*(w)*c_101 + \
                (u)*(v)*(1.0-w)*c_110 + \
                (u)*(v)*(w)*c_111
        return c_xyz


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))
        self.trilinear_interpolation = TrilinearIntepolation().cuda()

    def forward(self, xyz, importance=None, vq=None):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if vq is None:
            out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=False)
        else:
            out = self.trilinear_interpolation(self.grid, ind_norm.squeeze(0), vq)

        if importance is not None:
            sampled_importance = F.grid_sample(importance, ind_norm, mode='bilinear', align_corners=False)
            sampled_importance = sampled_importance.reshape(self.channels,-1).T.reshape(*shape,self.channels)
            if self.channels == 1:
                sampled_importance = sampled_importance.squeeze(-1)

        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        if importance is not None:
            return out, sampled_importance
        else:
            return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
'''
class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R+R+Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[...,[0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape,self.channels)
        else:
            out = compute_tensorf_val(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X,1], mode='bilinear', align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y,1], mode='bilinear', align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z,1], mode='bilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.x_vec[:,:,1:], self.x_vec[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.y_vec[:,:,1:], self.y_vec[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.z_vec[:,:,1:], self.z_vec[:,:,:-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([
                torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0,:,:,0]),
                torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0,:,:,0]),
                torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0,:,:,0]),
            ])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0,:,:,0]) + \
                   torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0,:,:,0]) + \
                   torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0,:,:,0])
            grid = grid[None,None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config["n_comp"]}'

def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = torch.cat([
        xy_feat * z_feat,
        xz_feat * y_feat,
        yz_feat * x_feat,
    ], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat

def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

