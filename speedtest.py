import time
from typing import Tuple

import numpy as np
import torch

from train.lednet import Net


def calculate_flops(model: torch.nn.Module, size: Tuple[int, ...] = (1, 3, 1024, 2048)):
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.eval()
    x = torch.randn(size).cuda()
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops, max_depth=1))


def benchmark(model: torch.nn.Module, size: Tuple[int, ...] = (1, 3, 1024, 2048), nwarmup: int = 50,
              nruns: int = 1000, verbose: bool = True):
    model.eval()
    x = torch.randn(size, device='cuda')

    if verbose:
        print('Warm up')
    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(x)
    torch.cuda.synchronize()
    if verbose:
        print('Benchmarking')
    timings = []
    with torch.no_grad():
        for _ in range(nruns):
            start_time = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    return np.mean(timings)


if __name__ == '__main__':
    model = Net(num_classes=19).cuda().eval()
    ds_list = [(256, 512), (384, 768), (512, 1024), (768, 1536), (1024, 2048)]
    for s in ds_list:
        print(s)
        t = 0
        calculate_flops(model, (1, 3, s[0], s[1]))
        for _ in range(5):
            t += benchmark(model, (1, 3, s[0], s[1]), verbose=False) / 5
        print(t)
