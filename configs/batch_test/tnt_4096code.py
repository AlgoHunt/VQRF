_base_ = '../default.py'

expname = 'dvgo_Truck'
basedir = './logs/tanks_and_temple'

data = dict(
    # datadir='/bfs/HoloResearch/NeRFData/TanksAndTemple/Truck',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)

