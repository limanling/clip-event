# Revised on https://raw.githubusercontent.com/pytorch/vision/master/references/detection/utils.py

from collections import defaultdict, deque
import datetime
import errno
import os
import time
import logging
import math
from bisect import bisect_right
from typing import List

import torch
import torch.distributed as dist
from datetime import timedelta
from utils_MPIAdapter import MPIAdapter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.9f} ({global_avg:.9f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# def all_gather(data):
#     """
#     Run all_gather on arbitrary picklable data (not necessarily tensors)
#     Args:
#         data: any picklable object
#     Returns:
#         list[data]: list of data gathered from each rank
#     """
#     world_size = get_world_size()
#     if world_size == 1:
#         return [data]
#     data_list = [None] * world_size
#     dist.all_gather_object(data_list, data)
#     return data_list
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = comm.world_size
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = comm.world_size
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

# # NOTE: change: average all ->> average only saved to rank 0
# def reduce_dict(input_dict, average=True):
#     """
#     Args:
#         input_dict (dict): all the values will be reduced
#         average (bool): whether to do average or sum
#     Reduce the values in the dictionary from all processes so that process with rank
#     0 has the averaged results. Returns a dict with the same fields as
#     input_dict, after reduction.
#     """
#     world_size = comm.world_size
#     if world_size < 2:
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#         values = torch.stack(values, dim=0)
#         dist.reduce(values, dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             values /= world_size
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict

# NOTE: new
def gather_tensors(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(comm.world_size)
    ]

    dist.all_gather(tensors_gather, tensor, async_op=False)
    # need to do this to restore propagation of the gradients
    tensors_gather[comm.rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logging.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


# def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

#     def f(x):
#         if x >= warmup_iters:
#             return 1
#         alpha = float(x) / warmup_iters
#         return warmup_factor * (1 - alpha) + alpha

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    # The code is borrowed from detectron2
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_epochs: int = 5,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    # The code is borrowed from detectron2
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_epochs: int = 5,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_epochs
        # instead of at 0. In the case that warmup_epochs << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_epochs: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_epochs (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_epochs:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_epochs
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise





def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0


def save_on_master(*args, **kwargs):
    if comm.is_main_process():
        torch.save(*args, **kwargs)


class Comm(object):
    def __init__(self, local_rank=0):
        self.local_rank = 0

    @property
    def world_size(self):
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @property
    def rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @property
    def local_rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return self._local_rank

    @local_rank.setter
    def local_rank(self, value):
        if not dist.is_available():
            self._local_rank = 0
        if not dist.is_initialized():
            self._local_rank = 0
        self._local_rank = value

    @property
    def head(self):
        return 'Rank[{}/{}]'.format(self.rank, self.world_size)
   
    def is_main_process(self):
        return self.rank == 0

    def synchronize(self):
        """
        Helper function to synchronize (barrier) among all processes when
        using distributed training
        """
        if self.world_size == 1:
            return
        dist.barrier()


comm = Comm()


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


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def init_distributed_mode(args):
    # # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # # args.distributed = num_gpus > 1

    # # if args.distributed:
    # port = str(_find_free_port())
    # mpi_adapter = MPIAdapter(port=port)
    # backend = 'nccl'
    # mpi_adapter.init_process_group(backend)
    # mpi_adapter.log_info()
    # args.dist_url = mpi_adapter.init_method_url
    # args.world_size = mpi_adapter.world_size
    # args.rank = mpi_adapter.rank
    # args.gpu = mpi_adapter.local_rank
    # torch.cuda.set_device(args.gpu)
    # args.distributed = True
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


    # print('os.environ', os.environ)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        if args.world_size < 2:
            print('Not using distributed mode since world size is %d' % args.world_size)
            args.distributed = False
            args.gpu = 0 if args.local_rank is None else args.local_rank
            return args    
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ['LOCAL_RANK']) # args.local_rank #
    elif 'SLURM_PROCID' in os.environ:
        print('distributed: SLURM_PROCID')
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.gpu = 0 if args.local_rank is None else args.local_rank
        return args

    args.distributed = True
    args.dist_backend = 'nccl'
    args.dist_url = "env://"
    # master_port = _find_free_port()
    # master_address = '127.0.0.1'
    # args.dist_url = f'tcp://{master_address}:{master_port}' #"env://"
    os.environ['NCCL_BLOCKING_WAIT'] = '1'

    torch.cuda.set_device(args.gpu)
    print('| distributed init: {}'.format( #(rank {}): {}'.format(
        # args.rank, 
        args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, #rank=args.rank,
                                         timeout=timedelta(minutes=60))
    comm.synchronize() # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


    # num_gpus = int(os.environ["WORLD_SIZE"]) \
    #     if "WORLD_SIZE" in os.environ else 1
    # args.cfg['distributed'] = num_gpus > 1

    # if args.cfg['distributed']:
    #     print("=> init process group start")
    #     os.environ['NCCL_BLOCKING_WAIT'] = '1'
    #     torch.cuda.set_device(args.cfg['local_rank'])
    #     torch.distributed.init_process_group(
    #         backend="nccl",
    #         init_method="env://",
    #         timeout=timedelta(minutes=60))
    #     comm.synchronize()
    #     comm.local_rank = args.cfg['local_rank']
    #     print("=> init process group end")

    return args

    