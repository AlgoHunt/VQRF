_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/materials',
    dataset_type='blender',
    white_bkgd=True,
)

