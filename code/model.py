import torch
import torch.nn as nn
import pdb


class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()

        self.dim = dim
        
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer


    def forward(self, x):

        B, T, C, H, W = x.shape

        feats = self.encoder(x.flatten(0,1)).reshape(B, T, self.dim)

        z1 = feats[:, :T//2,:]
        z2 = feats[:, T//2:,:]

        # # TODO: split into 2 clips after encoder passage
        # z1 = self.encoder(clip1.flatten(0,1)).reshape(B, T//2, self.dim)
        # z2 = self.encoder(clip2.flatten(0,1)).reshape(B, T//2, self.dim)

        # Clip Aggregation
        s1 = z1.mean(1)
        s2 = z2.mean(1)

        p1 = self.predictor(s1)
        p2 = self.predictor(s2)

        return p1, p2, s1.detach(), s2.detach()

