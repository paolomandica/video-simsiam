import torch
import torch.nn as nn
import pdb
from . import tcn

from copy import deepcopy

# import individual_TF


class TCN(nn.Module):
    def __init__(self, ):
        super(TCN, self).__init__()
        self.tcn = tcn.TemporalConvNet(input_size, num_channels,
                                       kernel_size=kernel_size, dropout=dropout)


class SimSiam(nn.Module):
    def __init__(self, base_encoder, aggregator, dim=2048,
                 pred_dim=512, device='cuda', n_frames=4):
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

        # Initialize aggregator
        self.aggregate = None
        if aggregator == "mean":
            self.aggregate = lambda x: x.mean(1)
        elif aggregator == "tcn":
            self.tcn = tcn.TemporalConvNet(n_frames//2, [1], kernel_size=dim)
            self.aggregate = lambda x: self.tcn(x).squeeze(1)

        # inp_enc_size =
        # inp_dec_size =
        # out_size =
        # tf_layers =
        # tf_emb_size =
        # tf_fc = 2048
        # tf_heads =
        # tf_dropout =
        # self.aggr=individual_TF.IndividualTF(2, 3, 3, N=tf_layers, d_model=tf_emb_size, d_ff=tf_fc, h=tf_heads, dropout=tf_dropout, mean=[0,0], std=[0,0]).to(self.device)

        # config= BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='relu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12)
        # model = BertModel(config).to(device)

        # a=NewEmbed(3, 768).to(device)
        # model.set_input_embeddings(a)
        # generator=GeneratorTS(768,2).to(device)

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

        s1 = self.aggregate(z1)
        s2 = self.aggregate(z2)

        ############## PREDICTOR ##############

        p1 = self.predictor(s1)
        p2 = self.predictor(s2)

        return p1, p2, s1.detach(), s2.detach()
