_base_ = '../default.py'

expname = 'lego_prune'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/bfs/HoloResearch/NeRFData/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)



fine_train = dict(
    N_iters=20000,
    importance_step=20000,
    prune_step=20000,
)

fine_model_and_render = dict(
    i_fully_vq=1000000,
)

vq_model_and_render = dict(
    codebook_size=4096,
)