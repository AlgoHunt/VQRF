import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss
import torch_scatter

import math

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    # parser.add_argument("--apply_quant", default=True, type=bool)
    
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    # vqrf options
    parser.add_argument("--fully_vq",  action="store_true",
                    help='fully vector quantize the full model')
    parser.add_argument("--init_importance",  action="store_true",
                help='initialize importance score')
    parser.add_argument("--importance_include",  type=float,  default=0.00,
            help='quantile threshold for non-vq-voxels')
    parser.add_argument("--importance_prune",  type=float,  default=1.0,
            help='quantile threshold for prune=voxels')
    parser.add_argument("--k_expire",  type=int,  default=10,
            help='expireed k code per iteration')
    parser.add_argument("--render_fine",  action="store_true", 
            help='rendering and testing the non compressed model')
            
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

        # break

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
        if eval_ssim and eval_lpips_vgg and eval_lpips_alex:
            np.savetxt(f'{savedir}/mean.txt', np.asarray([np.mean(psnrs), np.mean(ssims), np.mean(lpips_vgg), np.mean(lpips_alex)]))
        else:
            np.savetxt(f'{savedir}/mean.txt', np.asarray([np.mean(psnrs)]))

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def load_vqdvgo(path, device='cuda'):
    def load_f(name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path,name),allow_pickle=allow_pickle)[array_name]

    metadata =load_f('metadata.npz',allow_pickle=True,array_name='metadata')
    metadata = metadata.item()

    ## prepare needed model kwargs 
    grid_dequant = metadata.pop('grid_dequant')
    density_dequant = metadata.pop('density_dequant')
    model_kwargs = metadata['model_kwargs']
    codebook_size = model_kwargs['codebook_size']
    bit_length = int(math.log2(codebook_size))
    k0_dim  = model_kwargs['rgbnet_dim']
    world_size = metadata['world_size'].cpu().numpy().tolist()
    max_elements = metadata['world_size'].prod().item()

    ## loading the two masks  
    keep_mask = load_f('keep_mask.npz')
    non_prune_mask = load_f('non_prune_mask.npz') 

    keep_mask = np.unpackbits(keep_mask)
    non_prune_mask = np.unpackbits(non_prune_mask)

    keep_mask = keep_mask[:max_elements] #.reshape(world_size)
    non_prune_mask = non_prune_mask[:max_elements] #.reshape(world_size)

    keep_mask = torch.from_numpy(keep_mask).bool().to(device)
    non_prune_mask = torch.from_numpy(non_prune_mask).bool().to(device)

    ## loading codebook and vq indexes
    codebook = load_f('codebook.npz')
    codebook = torch.from_numpy(codebook).squeeze(0).float().to(device)
    vq_mask = torch.logical_xor(keep_mask,non_prune_mask) 
    vq_elements = vq_mask.sum()

    vq_indexs = load_f('vq_indexs.npz')
    vq_indexs = np.unpackbits(vq_indexs)
    vq_indexs = vq_indexs[:vq_elements*bit_length].reshape(vq_elements,bit_length)
    vq_indexs = torch.from_numpy(vq_indexs).float()
    vq_indexs = bin2dec(vq_indexs, bits=12)
    vq_indexs = vq_indexs.long().to(device)

    ## loading the non-vq-feature and non-prune density
    true_grid = load_f('non_vq_grid.npz')
    true_density = load_f('non_prune_density.npz')

    true_grid = (true_grid.astype(np.float32) - grid_dequant['zero_point'])*grid_dequant['scale']
    true_density = (true_density.astype(np.float32) - density_dequant['zero_point'])*density_dequant['scale']

    true_grid = torch.from_numpy(true_grid).float().to(device)
    true_density = torch.from_numpy(true_density).float().to(device)

    # build the actual feature and density grid
    full_grid = torch.zeros(max_elements, k0_dim).to(device)
    full_grid[keep_mask,:] = true_grid
    full_grid[vq_mask,:] = codebook[vq_indexs]

    full_density = torch.zeros(max_elements, 1).to(device) #- 99999
    full_density[non_prune_mask,:] = true_density

    mdoel_state_dict =  metadata['model_state_dict']
    rgbnet_npz = load_f('rgbnet.npz',allow_pickle=True)
    for k,v in rgbnet_npz.item().items():
        mdoel_state_dict['rgbnet.'+k] =v.to(device)
    mdoel_state_dict['k0.grid'] = full_grid.T.reshape(1,k0_dim,*world_size )
    mdoel_state_dict['density.grid'] = full_density.reshape(1,1,*world_size)
    return model_kwargs, mdoel_state_dict

if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)



    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    ckpt_name = 'extreme_last'
    model_kwargs, mdoel_state_dict = load_vqdvgo(os.path.join(cfg.basedir, cfg.expname,'extreme_saving'),device=device)
    model_kwargs['mask_cache_path'] = None
    model = model_class(**model_kwargs)
    model.eval()
    model.load_state_dict(mdoel_state_dict, strict=False)
    
    model.to(device)
    # model.mask_cache.mask &= (model.density.grid.squeeze(0).squeeze(0) != 0)
    model.mask_cache.mask[:] = True
    model.update_occupancy_cache()
    # import ipdb;ipdb.set_trace()


    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        },
    }

    

   
   
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

    print('Done')

