import datetime
import os
import time
import sys
import numpy as np
import json

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision

import data
from data.kinetics import Kinetics400
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler

import utils

from model import SimSiam

torch.autograd.set_detect_anomaly(True)

# Disable wandb syncing to the cloud
# os.environ['WANDB_MODE'] = 'offline'

####################################################################################################
# train_one_epoch function
####################################################################################################


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device,
                    epoch, print_freq, vis=None, checkpoint_fn=None):

    criterion = nn.CosineSimilarity(dim=1).cuda()

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = f'Epoch: [{epoch}]'

    # Initialise wandb
    if vis is not None:
        vis.wandb_init(model)

    for step, video in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        p1, p2, z1, z2 = model(video.to(device))
        loss = -(utils.matrix_cosine_similarity(p1, z2) +
                 utils.matrix_cosine_similarity(p2, z1)) * 0.5
        # loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        if vis is not None:
            vis.log(dict(loss=loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(
            video.shape[0] / (time.time() - start_time))
        # lr_scheduler.step()

        if np.random.rand() < 0.005:
            print("Save checkpoint...")
            checkpoint_fn()

    checkpoint_fn()


####################################################################################################
# Minor functions
# - _get_cache_path : get cache path for automatic caching of train dataset
# - collate_fn      : custom collate function for dataloader; removes audio from data samples
####################################################################################################


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio and labels from the batch
    batch = [d[0] for d in batch]
    return default_collate(batch)

####################################################################################################
# Main
####################################################################################################


def main(args):

    print("Arguments", end="\n" + "-"*100 + "\n")
    for arg, value in vars(args).items():
        print(f"{arg} = {value}")
    print("-"*100)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print("Preparing training dataloader", end="\n"+"-"*100+"\n")
    traindir = os.path.join(
        args.data_path, 'train_256' if not args.fast_test else 'val_256')
    valdir = os.path.join(args.data_path, 'val_256')

    st = time.time()
    cache_path = args.cache_path

    transform_train = utils.augs.get_train_transforms(args)

    # Dataset
    def make_dataset(is_train, cached=None):
        _transform = transform_train if is_train else transform_test

        return Kinetics400(
            traindir if is_train else valdir,
            frames_per_clip=args.clip_len,
            step_between_clips=args.clips_step,
            transform=transform_train,
            extensions=('mp4'),
            frame_rate=args.frame_skip,
            # cached=cached,
            _precomputed_metadata=cached
        )

    if args.cache_dataset and os.path.exists(cache_path):
        print(
            f"Loading dataset_train from {cache_path}", end="\n"+"-"*100+"\n")
        dataset, _ = torch.load(cache_path)
        cached = dict(video_paths=dataset.video_clips.video_paths,
                      video_fps=dataset.video_clips.video_fps,
                      video_pts=dataset.video_clips.video_pts)

        dataset = make_dataset(
            is_train=True, cached=cached)
        dataset.transform = transform_train
    else:
        dataset = make_dataset(is_train=True)
        if 'kinetics' in args.data_path.lower():  # args.cache_dataset and
            print(
                f"Saving dataset_train to {cache_path}", end="\n"+"-"*100+"\n")
            utils.mkdir(os.path.dirname(cache_path))
            dataset.transform = None
            torch.save((dataset, traindir), cache_path)
            dataset.transform = transform_train

    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(
            args.clip_len, 1, frame_rate=args.frame_skip)

    print("Took", time.time() - st)

    # Data Loader
    def make_data_sampler(is_train, dataset):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler  # UniformClipSampler
            return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None

    print("Creating data loaders", end="\n"+"-"*100+"\n")
    train_sampler = make_data_sampler(True, dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,  # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=args.workers//2,
        pin_memory=True, collate_fn=collate_fn)

    # print("Set Compactness at:", args.compactness)
    # data_loader.dataset.set_compactness(args.compactness)

    # Visualisation
    vis = utils.visualize.Visualize(args) if args.visualize else None

    # Model
    model = SimSiam(utils.get_ResNet(), device=device).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    # Learning rate schedule
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = None
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    model_without_ddp = model

    # Parallelise model over GPUs
    if args.data_parallel:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module

    # Partially load weights from model checkpoint
    if args.partial_reload:
        checkpoint = torch.load(args.partial_reload, map_location='cpu')
        utils.partial_load(checkpoint['model'], model_without_ddp)
        optimizer.param_groups[0]["lr"] = args.lr
        # args.start_epoch = checkpoint['epoch'] + 1

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    def save_model_checkpoint():
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),  # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f'model_{epoch}.pth'))
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    # Start Training
    print("Start training", end="\n"+"-"*100+"\n")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq,
                        vis=vis, checkpoint_fn=save_model_checkpoint)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

####################################################################################################
# Run as Script
####################################################################################################


if __name__ == "__main__":
    args = utils.arguments.train_args()
    main(args)
