import sys
import os
import argparse
from multiprocessing import Process, Queue
from typing import List, Dict
import subprocess
import mmcv

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", "-g", type=str, required=True,
                            help="space delimited GPU id list (global id in nvidia-smi, "
                                 "not considering CUDA_VISIBLE_DEVICES)")
parser.add_argument('--eval', action='store_true', default=False,
                   help='evaluation mode (run the render_imgs script)')
parser.add_argument('--configname',  default='baseline',
                   help='a.k.a config  in subfloer ./configs/batchtest/<configname>.txt ')

parser.add_argument("--importance_include",  type=float,  default=0.6,
        help='fully vector quantize the full model')

parser.add_argument("--importance_prune",  type=float,  default=0.999,
        help='fully vector quantize the full model')

parser.add_argument("--codebook_size",  type=int,  default=4096,
        help='fully vector quantize the full model')

parser.add_argument("--dump_images",  action='store_true', default=False,
        help='fully vector quantize the full model')

parser.add_argument('--dataset', type=str, default='syn', choices=['syn', 'llff', 'tnt', 'nsvf'])
args = parser.parse_args()

PSNR_FILE_NAME = 'test_psnr.txt'

def run_exp(env,  config, datadir, expname, basedir):
    
  
    psnr_file_threestep_path = os.path.join(basedir, expname,'render_test_vq_last','mean.txt' )
    psnr_file_path = os.path.join(basedir, expname,'render_test_fine_last','mean.txt' )
    cfg = mmcv.Config.fromfile(config)
    cfg.expname = expname
    cfg.data.datadir = datadir
    cfg.basedir = basedir
    cfg.vq_model_and_render.codebook_size = args.codebook_size
    
    auto_config_path = f'./configs/auto/{expname}.py'
    cfg.dump(auto_config_path)
    print('********************************************')
    
    
    base_cmd = ['python', 'run_final.py',  '--config', auto_config_path, '--eval_ssim','--eval_lpips_vgg',
            '--eval_lpips_alex' , '--render_test',  '--fully_vq --render_fine ', 
            f'--importance_prune {args.importance_prune}' ,f'--importance_include {args.importance_include}']
    if os.path.isfile(psnr_file_path) or  os.path.isfile(psnr_file_threestep_path): 
        base_cmd.append('--render_only')
        # pass
    if args.dump_images:
        base_cmd.append('--dump_images')

    opt_cmd = ' '.join(base_cmd)
    print(opt_cmd, "on ", env["CUDA_VISIBLE_DEVICES"])
    opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
        sys.stdout.encoding)




def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
      
        run_exp(env, **task)


DatasetSetting={
    "syn": {
        "data": "./data/nerf_synthetic",
        "cfg": f"./configs/batch_test/{args.configname}.py",
        "basedir":f"./logs/{args.configname}",
        "scene_list":['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    },
    "tnt":{
        "data": "./data/TanksAndTemple",
        "cfg": f"./configs/batch_test/{args.configname}.py",
        "basedir":f"./logs/{args.configname}",
        "scene_list":['Barn', 'Caterpillar', 'Family', 'Ignatius', 'Truck']
    },
    "nsvf":{
        "data": "./data/Synthetic_NSVF",
        "cfg": f"./configs/batch_test/{args.configname}.py",
        "basedir":f"./logs/{args.configname}",
        "scene_list":['Bike', 'Lifestyle', 'Palace', 'Robot', 'Spaceship', 'Steamtrain', 'Toad', 'Wineholder',]
    }
    
}

datasetting = DatasetSetting[args.dataset]
all_tasks = []

for scene in datasetting["scene_list"]:
    task: Dict = {}
    task['datadir'] = f'{datasetting["data"]}/{scene}'
    task['expname'] = f'{args.configname}_{scene}'  
    task["config"] = datasetting['cfg']
    task["basedir"] = datasetting["basedir"]
    assert os.path.exists(task['datadir']), task['datadir'] + ' does not exist'
    assert os.path.isfile(task['config']), task['config'] + ' does not exist'
    all_tasks.append(task)

pqueue = Queue()
for task in all_tasks:
    pqueue.put(task)

args.gpus = list(map(int, args.gpus.split()))
print('GPUS:', args.gpus)

for _ in args.gpus:
    pqueue.put({})

all_procs = []
for i, gpu in enumerate(args.gpus):
    process = Process(target=process_main, args=(gpu, pqueue))
    process.daemon = True
    process.start()
    all_procs.append(process)

for i, gpu in enumerate(args.gpus):
    all_procs[i].join()



class AverageMeter(object):
    def __init__(self, name=''):
        self.name=name
        self.reset()
    def reset(self):
        self.val=0
        self.sum=0
        self.avg=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg=self.sum/self.count
    def __repr__(self) -> str:
        return f'{self.name}: average {self.count}: {self.avg}\n'

from prettytable import PrettyTable
table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS_ALEX","LPIPS_VGG", "SIZE"])
table.float_format = '.3'


PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS_A=AverageMeter('LPIPS_A')
LPIPS_V=AverageMeter('LPIPS_V')
SIZE=AverageMeter('SIZE')


for scene in datasetting["scene_list"]: #[ 'chair', 'drums', 'ficus', 'hotdog', 'lego', 'mic', 'materials', 'ship'   ]:

    path = f'./logs/{args.configname}/{args.configname}_{scene}/render_test_vq_last/mean.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        psnr = float(lines[0].strip())
        ssim = float(lines[1].strip())
        lpips_a = float(lines[2].strip())
        lpips_v = float(lines[3].strip())
        PSNR.update(psnr)
        SSIM.update(ssim)
        LPIPS_A.update(lpips_a)
        LPIPS_V.update(lpips_v)
        compressed_file = f'./logs/{args.configname}/{args.configname}_{scene}/extreme_saving.zip'
        if os.path.exists(compressed_file):
            size = os.path.getsize(compressed_file)/(1024*1024)
        else:
            size = 0
        table.add_row([scene, psnr, ssim, lpips_a, lpips_v, size])
        SIZE.update(size)
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS_A.avg,LPIPS_V.avg, SIZE.avg])

txt_file = os.path.join(datasetting["basedir"], f'merge_{args.importance_include}_{args.importance_prune}_{args.codebook_size}.txt')
with open(txt_file, 'w') as f:
    f.writelines(table.get_string())

csv_file = os.path.join(datasetting["basedir"], f'merge_{args.importance_include}_{args.importance_prune}_{args.codebook_size}.csv')
with open(csv_file, 'w', newline='') as f:
    f.writelines(table.get_csv_string())
print('VQ-DVGO:')
print(table)


table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS_ALEX","LPIPS_VGG", "SIZE"])
table.float_format = '.3'

PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS_A=AverageMeter('LPIPS_A')
LPIPS_V=AverageMeter('LPIPS_V')
SIZE=AverageMeter('SIZE')


for scene in datasetting["scene_list"]: #[ 'chair', 'drums', 'ficus', 'hotdog', 'lego', 'mic', 'materials', 'ship'   ]:
    path = f'./logs/{args.configname}/{args.configname}_{scene}/render_test_fine_last/mean.txt'

    with open(path, 'r') as f:
        lines = f.readlines()
        psnr = float(lines[0].strip())
        ssim = float(lines[1].strip())
        lpips_a = float(lines[2].strip())
        lpips_v = float(lines[3].strip())
        PSNR.update(psnr)
        SSIM.update(ssim)
        LPIPS_A.update(lpips_a)
        LPIPS_V.update(lpips_v)
        compressed_file = f'./logs/{args.configname}/{args.configname}_{scene}/fine_model.pth.zip'
        if os.path.exists(compressed_file):
            size = os.path.getsize(compressed_file)/(1024*1024)
        else:
            size = 0
        table.add_row([scene, psnr, ssim, lpips_a, lpips_v, size])
        SIZE.update(size)
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS_A.avg,LPIPS_V.avg, SIZE.avg])

txt_file = os.path.join(datasetting["basedir"], f'merge_fine.txt')
with open(txt_file, 'w') as f:
    f.writelines(table.get_string())

csv_file = os.path.join(datasetting["basedir"], f'merge_fine.csv')
with open(csv_file, 'w', newline='') as f:
    f.writelines(table.get_csv_string())
print('orginal DVGO:')
print(table)