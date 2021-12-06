import torch
import torch.nn as nn
import pdb

from copy import deepcopy


class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512, aggr_hidden=2048, 
                aggr_layers=1, aggr_directions=1, device='cuda'):
        super(SimSiam, self).__init__()

        self.dim = dim
        self.device = device

        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        #Initialize aggregator
        self.aggr_hidden = aggr_hidden
        self.aggr_layers = aggr_layers
        self.aggr_directions = aggr_directions
        bidirectional = True if aggr_directions==2 else False
        
        self.lstm = nn.LSTM(dim, aggr_hidden, aggr_layers, batch_first=True, bidirectional=bidirectional)
        self.lstm_linear = nn.Linear(aggr_hidden, dim)

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x):

        B, T, C, H, W = x.shape


        ############## FRAMES ENCODING ##############
        feats = self.encoder(x.flatten(0, 1)).reshape(B, T, self.dim)

        z1 = feats[:, :T//2, :]
        z2 = feats[:, T//2:, :]


        ############## FRAMES AGGREGATION ##############

        ### MEAN ###
        # s1 = z1.mean(1)
        # s2 = z2.mean(1)

        ### LSTM ###
        h_t = torch.zeros(self.aggr_layers*self.aggr_directions, B, self.aggr_hidden) # num_layers*num_directions, batch, hidden_size
        c_t = torch.zeros(self.aggr_layers*self.aggr_directions, B, self.aggr_hidden)

        hidden_z1 = (h_t.to(self.device), c_t.to(self.device))
        hidden_z2 = deepcopy(hidden_z1)

        z1, hidden_z1 = self.lstm(z1, hidden_z1)
        z2, hidden_z2 = self.lstm(z2, hidden_z2)

        s1 = self.lstm_linear(hidden_z1[0].squeeze(0))
        s2 = self.lstm_linear(hidden_z2[0].squeeze(0))

        ############## PREDICTOR ##############

        p1 = self.predictor(s1)
        p2 = self.predictor(s2)

        return p1, p2, s1.detach(), s2.detach()
