import torch
import torch.nn.functional as F


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name, debug=False):
        self.name = name
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        if not self.debug:
            return

        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")

@torch.no_grad()
def unshuffle(x, downscale_factor=2):
    '''
    Args:
        x: [N, H, W, C]
            or
        x: [H, W, C]
    
    Ret:
        [N, W//fac, H//fac, C, fac*fac]
            or
        [W//fac, H//fac, C, fac*fac]
    '''
    if len(x.shape) == 4:
        x = x.permute(0,3,1,2).unsqueeze(2)
        x = F.pixel_unshuffle(x, downscale_factor=downscale_factor).permute(0,3,4,1,2)
    elif len(x.shape) == 3:
        x = x.permute(2,0,1).unsqueeze(1)
        x = F.pixel_unshuffle(x, downscale_factor=downscale_factor).permute(2,3,0,1)
    else:
        raise NotImplementedError
    return x

@torch.no_grad()
def shuffle(x, upscale_factor=2):
    '''
    Args:
        [N, W//fac, H//fac, C, fac*fac]
            or
        [W//fac, H//fac, C, fac*fac]
    
    Ret:
        [N,  H, W, C]
            or
        [H, W, C]
    '''
    if len(x.shape) == 5:
        x = x.permute(0,3,4,1,2)
        x = F.pixel_shuffle(x, upscale_factor=upscale_factor).squeeze(2).permute(0,2,3,1)
    elif len(x.shape) == 4:
        x = x.permute(2,3,0,1)
        x = F.pixel_shuffle(x, upscale_factor=upscale_factor).squeeze(1).permute(1,2,0)
       
    else:
        raise NotImplementedError
    return x