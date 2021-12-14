from torchvision import transforms
from torch.nn import functional as F
from torch import nn
from torchvision import models
from collections import defaultdict, deque
import datetime
import time
import torch

import errno
import os
import sys
import numbers

from . import arguments
from . import visualize
from . import augs

#########################################################
# DEBUG
#########################################################


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"


sys.excepthook = info

#########################################################
# Meters
#########################################################


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        import torch.distributed as dist
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
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
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#################################################################################
# Network Utils
#################################################################################


def get_ResNet():
    return models.__dict__['resnet18']


def partial_load(pretrained_dict, model, skip_keys=[]):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items(
    ) if k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('\nSkipped keys: ', skipped_keys)
    print('\nLoading keys: ', filtered_dict.keys())


class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''

    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet

    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        m = self.model(xx)

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)


class ZeroSoftmax(nn.Module):
    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class MaskedAttention(nn.Module):
    '''
    A module that implements masked attention based on spatial locality 
    TODO implement in a more efficient way (torch sparse or correlation filter)
    '''

    def __init__(self, radius, flat=True):
        super(MaskedAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.index = {}

    def mask(self, H, W):
        if not ('%s-%s' % (H, W) in self.masks):
            self.make(H, W)
        return self.masks['%s-%s' % (H, W)]

    def index(self, H, W):
        if not ('%s-%s' % (H, W) in self.index):
            self.make_index(H, W)
        return self.index['%s-%s' % (H, W)]

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)

        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ((gx[None, None, :, :] - gx[:, :, None, None])**2 +
             (gy[None, None, :, :] - gy[:, :, None, None])**2).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = self.flatten(D)
        self.masks['%s-%s' % (H, W)] = D

        return D

    def flatten(self, D):
        return torch.flatten(torch.flatten(D, 1, 2), -2, -1)

    def make_index(self, H, W, pad=False):
        mask = self.mask(H, W).view(1, -1).byte()
        idx = torch.arange(0, mask.numel())[mask[0]][None]

        self.index['%s-%s' % (H, W)] = idx

        return idx

    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H, W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]

        return x * mask[0]


def matrix_cosine_similarity(x1, x2):
    B, N, C = x1.shape

    # normalize tensors
    x1 = torch.nn.functional.normalize(x1, dim=-1)
    x2 = torch.nn.functional.normalize(x2, dim=-1)

    # compute matrix multiplication and remove diagonal
    mat_sim = torch.einsum('bnc,bmc->bnm', x1, x2)
    device = 'cuda:' + str(mat_sim.get_device())
    mat_sim_no_diag = mat_sim - torch.eye(N, N).to(device)

    # compute loss
    loss = mat_sim_no_diag.sum(-1).sum(-1) / (N*(N-1))

    return loss.mean()

#########################################################
# Misc
#########################################################


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img
