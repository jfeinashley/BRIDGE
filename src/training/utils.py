"""
Training utilities including metrics, schedulers, and helpers
"""

import torch
import torch.distributed as dist
import os
import math
from typing import Dict, Any, List
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.maxlen = window_size
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.maxlen:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return np.median(self.deque) if self.deque else 0

    @property
    def avg(self):
        return np.mean(self.deque) if self.deque else 0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value)


class MetricLogger:
    """Metric logger for training stats"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        import sys
        from itertools import islice
        
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        log_msg = self.delimiter.join([
            header,
            'Iter: [{0}/{1}]',
            'Time: {batch_time} ({batch_time_avg})',
            'Data: {data_time} ({data_time_avg})',
            '{meters}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                meters = str(self.delimiter.join([str(v) for k, v in self.meters.items()]))
                print(log_msg.format(
                    i, len(iterable),
                    batch_time=iter_time.value,
                    batch_time_avg=str(iter_time),
                    data_time=data_time.value,
                    data_time_avg=str(data_time),
                    meters=meters
                ))
                sys.stdout.flush()
            i += 1
            end = time.time()

    def synchronize_between_processes(self):
        """
        Synchronize metrics between processes in distributed training
        """
        if not is_dist_avail_and_initialized():
            return
        for meter in self.meters.values():
            if hasattr(meter, 'count') and hasattr(meter, 'total'):
                t = torch.tensor([meter.count, meter.total], dtype=torch.float64, device='cuda')
                dist.barrier()
                dist.all_reduce(t)
                t = t.tolist()
                meter.count = int(t[0])
                meter.total = t[1]

    @property
    def global_avg(self):
        """Get global average of all meters"""
        return {k: meter.global_avg for k, meter in self.meters.items()}


def warmup_lr_schedule(optimizer, step, warmup_steps, warmup_lr, init_lr):
    """Warmup learning rate schedule"""
    if step < warmup_steps:
        lr = warmup_lr + (init_lr - warmup_lr) * step / warmup_steps
    else:
        lr = init_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Cosine learning rate schedule"""
    lr = min_lr + (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Step learning rate schedule"""
    lr = max(init_lr * (decay_rate ** epoch), min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Distributed training utilities
def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank for distributed training"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0


def init_distributed_mode(args):
    """Initialize distributed training mode"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()


def save_on_master(*args, **kwargs):
    """Save checkpoint only on master process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# Import time for the functions
import time
