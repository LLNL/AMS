'''
based on: 
https://lc.llnl.gov/gitlab/autonomous-multiscale-project/ams/-/blob/main/ml/models/CHEETAH_Representation_Learning/modeldefs.py?ref_type=heads
'''

import torchvision.transforms as transforms
import random
import numpy as np
from math import pi
from math import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

class CheetahSurrogateDeltaUQ(nn.Module):
    def __init__(self, num_inputs=8, num_classes=7, initial_state=None):
        super().__init__()
        config = {}
        config['learning_rate'] = 1e-4
        #config['learning_rate'] = 1e-6
        config['n_epochs'] = 200
        #config['n_layers'] = 6
        config['n_layers'] = 4
        config['hidden_dim'] = 128
        config['dist'] = 'uniform'
        config['mapping_size'] = 128
        #config['batch_size'] = 1000
        config['batch_size'] = 100
        config['activation'] = 'relu'
        config['dropout'] = True
        config['dropout_prob'] = 0.00001
        config['batchnorm'] = False
        config['delta_uq'] = self.uq = True
        self.pe = 'IPT'
        if self.pe == 'PLE' or self.pe == 'NE':
            std = 0
        else:
            std = 2**-3
        config['variance'] = std**2
        self.dim_inp = num_inputs
        self.dim_out = num_classes
        self.config = config

        # Positional Embedding Definition
        scale = torch.sqrt(torch.tensor(self.config['variance']))
        self.ipt = IPTModule(self.dim_inp, self.config['mapping_size'], scale) #Basis function
        self.dim_inp = self.config['mapping_size']*2

        if self.uq:
            self.dim_inp *= 2

        # Surrogate Network Definition
        n = MLP(self.config, self.dim_inp, self.dim_out)
        fc_layers, last_layer = n.get_layer_chunks()
        fc_layers = nn.Sequential(*fc_layers)

        #self.opt = Adam(list(self.net.parameters())+list(self.ipt.parameters()), lr=self.config['learning_rate'])
        self.opt = Adam(list(n.parameters())+list(self.ipt.parameters()), lr=self.config['learning_rate'])

        #NOTE: UQ network parameters being learnable seems to be a no no.
        if self.uq:
            #self.net = deltaUQ_MLP(base_network=n)
            # need to break these layer chunks up for BADGE - brian
            self.DUQnet = deltaUQ_MLP(base_network=n, pre_last_layer=fc_layers, last_layer=last_layer) 

        if initial_state:
            self.initial_state = initial_state
            self.load_state_dict(torch.load(initial_state, map_location=torch.device("cpu")))
            print(f'Initialized model with values from {initial_state}')
            initial_opt_state = initial_state.replace('weights/weight_', 'opt_states/opt_state_')
            self.opt.load_state_dict(torch.load(initial_opt_state, map_location=torch.device("cpu")))
            print(f'Initialized optimizer with values from {initial_opt_state}')
            
    def get_embedding_dim(self):
        return self.config['hidden_dim']
    
    def forward(self, x, last=False, freeze=False):
        x = self.ipt(x)
        return self.DUQnet(x, last=last, freeze=freeze)
        

class deltaUQ(torch.nn.Module):
    def __init__(self, base_network=None, pre_last_layer=None, last_layer=None):
        super(deltaUQ, self).__init__()
        self.net = base_network
        self.pre_last_layer = pre_last_layer
        self.last_layer = last_layer

    def create_anchored_batch(self,x,anchors=None,n_anchors=1,corrupt=False):
        '''
        anchors (default=None):
            if passed, will use the same set of anchors for all batches during training.
            if  None, we will use a shuffled input minibatch to forward( ) as anchors for that batch (random)

            During *inference* it is recommended to keep the anchors fixed for all samples.

            n_anchors is chosen as min(n_batch,n_anchors)
        '''
        n_img = x.shape[0]
        if anchors is None:
            anchors = x[torch.randperm(n_img),:]
        
        ## make anchors (n_anchors) --> n_img*n_anchors
        if self.training:
            A = anchors[torch.randint(anchors.shape[0],(n_img*n_anchors,)),:]
        else:
            A = torch.repeat_interleave(anchors[torch.randperm(n_anchors),:],n_img,dim=0)    

        if corrupt:
            refs = self.corruption(A)
        else:
            refs = A

        ## before computing residual, make minibatch (n_img) --> n_img* n_anchors
        if len(x.shape)<=2:

            diff = x.tile((n_anchors,1))
            assert diff.shape[1]==A.shape[1], f"Tensor sizes for `diff`({diff.shape}) and `anchors` ({A.shape}) don't match!"
            diff -= A
        else:
            diff = x.tile((n_anchors,1,1,1)) - A

        batch = torch.cat([refs,diff],axis=1)
        return batch

    def corruption(self,samples):
        #base case does not use corruption in anchoring
        return samples

class deltaUQ_MLP(deltaUQ):
    def forward(self, x,  last=False, freeze=False, 
                anchors=None,corrupt=False,n_anchors=1,return_std=False):
        # no calibration
        if n_anchors==1 and return_std:
            raise Warning('Use n_anchor>1, std. dev cannot be computed!')

        a_batch = self.create_anchored_batch(x,anchors=anchors,n_anchors=n_anchors,corrupt=corrupt)
        #p = self.net(a_batch)
        if freeze:
            with torch.no_grad():
                penultimate = self.pre_last_layer(a_batch)
        else:
            penultimate = self.pre_last_layer(a_batch)

        p = self.last_layer(penultimate)
        p = p.reshape(n_anchors,x.shape[0],p.shape[1])
        p = p.mean(0)
        if return_std:
            std = p.std(0)

        if last:
            return p, penultimate
        else:
            if return_std:
                return p, std
            return p
        
class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(MLP, self).__init__()
        self.config = config
        self.layers = [MLPLayer(self.config['activation'], input_dim, self.config['hidden_dim'], do=False, dop=0.0, bn=False, is_first=True, is_last=False)]

        for i in range(1, self.config['n_layers'] - 1):
            self.layers.append(MLPLayer(self.config['activation'], self.config['hidden_dim'], self.config['hidden_dim'], do=self.config['dropout'], dop=self.config['dropout_prob'], bn=self.config['batchnorm'], is_first=False, is_last=False))
        self.layers.append(MLPLayer('identity', self.config['hidden_dim'], output_dim, do=False, dop=0.0, bn=False, is_first=False, is_last=True))

        self.mlp = nn.Sequential(*self.layers)
        self.relu = nn.ReLU()
        
    def get_layer_chunks(self):
        return self.layers[:-1], self.layers[-1]

    def forward(self,x):
        out = self.mlp(x)
        return out

class MLPLayer(nn.Module):
    def __init__(self, activation, input_dim, output_dim, do=True, dop=0.3, bn=True, is_first=False, is_last=False):
        super().__init__()

        self.do = do
        self.bn = bn
        self.is_first = is_first
        self.is_last = is_last

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'identity':
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError("Only 'relu', 'tanh' and 'identity' activations are supported")
        self.linear = nn.Linear(input_dim, output_dim)
        if self.do:
            self.dropout = nn.Dropout(dop)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        if self.is_first or self.is_last:
            return x
        else:
            if self.bn:
                x = self.batchnorm(x)
            if self.do:
                x = self.dropout(x)
            return x

class IPTModule(nn.Module):
    def __init__(self,input_dim, m, scale):
        super().__init__()
        self.Wr = torch.nn.Linear(input_dim, m) #Learnable Fourier basis
        with torch.no_grad():
            self.Wr.weight.normal_(0, scale)
        self.Wp = torch.nn.Linear(2*m, 2*m)

    def input_mapping(self,x):
        x_proj = (2. * pi * x) @ self.Wr.t()
        return

    def forward(self,x):
        h = torch.cat([torch.sin(2*pi*self.Wr(x)), torch.cos(2*pi*self.Wr(x))], dim=-1)
        out = self.Wp(h)
        return out