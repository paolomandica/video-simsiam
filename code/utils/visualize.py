# Video's features
import wandb
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.segmentation import mark_boundaries
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        video = (video*255).astype(np.uint8)

    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration=0.2)


class Visualize(object):
    def __init__(self, args):

        self._env_name = args.name
        # self.vis = visdom.Visdom(
        #     port=args.port,
        #     server='http://%s' % args.server,
        #     env=self._env_name,
        # )
        self.args = args

        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project="clip_simsiam",
                       entity="sapienzavideocontrastive",
                       group="release",
                       config=self.args)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)

    def save(self):
        self.vis.save([self._env_name])
