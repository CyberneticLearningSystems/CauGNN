import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from layers import GraphAttentionLayer, SpGraphAttentionLayer
# from torch_geometric.nn import GCNConv
# from ... import layers
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'TENet_master'))
from layer import DeGINConv,DenseGCNConv,DenseSAGEConv,DenseGraphConv,rkGraphConv
from torch_geometric.nn import GATConv

class Model(nn.Module):

    def __init__(self, args, A: np.ndarray):
        super(Model,self).__init__()

        # set parameters from args
        #! No clue what skip_mode does, but I've added it to the args as "concat" --> it's unused here
        self.skip_mode: str = args.skip_mode
        self.BATCH_SIZE: int = args.batch_size
        self.dropout: float = args.dropout
        self.use_cuda: bool = args.cuda
        self.n_e: int = args.n_e
        self.decoder: str = args.decoder
        self.attention_mode: str = args.attention_mode
        self.num_adjs: int = args.num_adj

        self._set_A(A)
        if self.num_adjs > 1:
            self._set_adjs(args.B)


        #! No clue what the following code is for, but I've added a num_adj parameter to the args and set it to 1 to skip this
        #* I assume that adjs is the adjacency matrix for the graph, not sure why there would be multiple though.
        self.adjs: list[np.ndarray] = [self.A]
        if self.num_adjs>1:
            #! Typing in this if block is a bit of a mess (in the name of saving memory)
            # I believe they're setting this using A and A_new to save memory
            A = np.loadtxt(args.B)
            A = np.array(A, dtype=np.float32)
            # divide A by the sum over axis=1 --> why different than for A?
            A = A / np.sum(A, 1)
            A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
            for i in range(args.batch_size):
                A_new[i, :, :] = A
            
            #! this is a temporary fix
            try:
                self.B = torch.from_numpy(A_new).cuda()
            except:
                self.B = torch.from_numpy(A_new).cpu()

            # I believe they're setting this using A and A_new to save memory
            A = np.ones((args.n_e,args.n_e),np.int8)
            # divide A by the sum over axis=1 --> why different than for A?
            A = A / np.sum(A, 1)
            A_new = np.zeros((args.batch_size, args.n_e, args.n_e), dtype=np.float32)
            for i in range(args.batch_size):
                A_new[i, :, :] = A
            self.C = torch.from_numpy(A_new).cuda()
            self.adjs = [self.A,self.B,self.C]

        
        ## The hyper-parameters are applied to all datasets in all horizons
        #* As there is a 1 in each Convolutional layer, the input size is the same for each layer (they are not stacked)
        #* FEATURE EXTRACTION LAYERS
        self.conv1 = nn.Conv2d(1, args.channel_size, kernel_size = (1, args.k_size[0]), stride=1)
        self.conv2 = nn.Conv2d(1, args.channel_size, kernel_size = (1, args.k_size[1]), stride=1)
        self.conv3 = nn.Conv2d(1, args.channel_size, kernel_size = (1, args.k_size[2]), stride=1)

        #? what are these maxpool layers used for? Can we remove them?
        # self.maxpool1 = nn.MaxPool2d(kernel_size = (1,args.k_size[0]),stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1, args.k_size[1]), stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1, args.k_size[2]), stride=1)
        # self.dropout = nn.Dropout(p=0.1)

        d = (len(args.k_size)*(args.window) - sum(args.k_size) + len(args.k_size))*args.channel_size #* This is the lenght of the input to the GCN
        

        #* DECODER LAYERS
        if self.decoder == 'GCN':
            # https://arxiv.org/pdf/1609.02907.pdf (Semi-Supervised Classification with Graph Convolutional Networks)
            #GCN Graph Network reduces the dimensionality of the input to 1 feature per node 

            self.gcn1 = DenseGCNConv(d, args.hid1)
            self.gcn2 = DenseGCNConv(args.hid1, args.hid2)
            self.gcn3 = DenseGCNConv(args.hid2, 1)

        if self.decoder == 'GNN':
            # https://arxiv.org/pdf/1810.00826v3.pdf (Graph Neural Networks)
            # self.gnn0 = DenseGraphConv(d, h0)
            self.gnn1 = DenseGraphConv(d, args.hid1)
            self.gnn2 = DenseGraphConv(args.hid1, args.hid2)
            self.gnn3 = DenseGraphConv(args.hid2, 1)

        if self.decoder == 'rGNN':
            # https://arxiv.org/pdf/1706.02216.pdf (Relational Graph Convolutional Networks)
            self.gc1 = rkGraphConv(self.num_adjs,d,args.hid1,self.attention_mode,aggr='mean')
            self.gc2 = rkGraphConv(self.num_adjs,args.hid1,args.hid2,self.attention_mode,aggr='mean')
            self.gc3 = rkGraphConv(self.num_adjs,args.hid2, 1, self.attention_mode, aggr='mean')

        if self.decoder == 'SAGE':
            # https://arxiv.org/pdf/1706.02216.pdf (SAmple and aggreGatE)
            self.sage1 = DenseSAGEConv(d,args.hid1)
            self.sage2 = DenseSAGEConv(args.hid1, args.hid2)
            self.sage3 = DenseSAGEConv(args.hid2, 1)

        if self.decoder == 'GIN':
            # https://arxiv.org/pdf/1810.00826v3.pdf (Graph Isomorphism Network)
            ginnn = nn.Sequential(
                nn.Linear(d,args.hid1),
                #! why is the second hidden layer skipped for GIN?
                # nn.ReLU(True),
                # nn.Linear(args.hid1, args.hid2),
                nn.ReLU(True),
                nn.Linear(args.hid1,1),
                nn.ReLU(True)
            )
            self.gin = DeGINConv(ginnn)
        if self.decoder == 'GAT':
            # https://arxiv.org/pdf/1710.10903.pdf (Graph Attention Networks)
            self.gatconv1 = GATConv(d,args.hid1)
            self.gatconv2 = GATConv(args.hid1,args.hid2)
            self.gatconv3 = GATConv(args.hid2,1)

        # Highway networks improve information transfer across layers, I think it might be this paper: https://paperswithcode.com/method/highway-networks
        self.hw = args.highway_window
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)


    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1
    
    
    #* when the class is called like a function and passed an input, this function is called (inherited from nn.Module)
    def forward(self, x: torch.Tensor):
        c=x.permute(0,2,1) #x: batch_size x window_size x features --> c: batch_size x features x window_size
        c=c.unsqueeze(1) #batch_size x 1 x features x window --> 1 is the height of the image (1D convolution)

        # if self.decoder != 'GAT':
        a1=self.conv1(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1) #Ouput conv(c): batch_size, num_filters, width (n features), height  --> permutate it to: batch_size, width, num_filters, height --> reshape it to: batch_size, width, num_filters*height
        a2=self.conv2(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1) 
        a3=self.conv3(c).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        # TODO: use these layers when self.dropout > 0
        # a1 = self.dropout(a1)
        # a2 = self.dropout(a2)
        # a3 = self.dropout(a3)

        #? what 
        # x_conv = F.relu(torch.cat([a1,a2],2))
        x_conv = F.relu(torch.cat([a1, a2, a3], 2)) #Stacks the outputs of the convolutional layers 
        # x_conv=F.relu(torch.cat([a1,a2,a3,a4,a5],2))
        # print(x_conv.shape)

        #? I think this can be removed
        ##GCN1
        # x1=F.relu(torch.bmm(self.A,x_conv).bmm(self.w1))
        # x2=F.relu(torch.bmm(self.A,x1).bmm(self.w2))
        # x3=F.relu(torch.bmm(self.A,x2).bmm(self.w3))

        # TODO: simplify this code to be more readable? (potentially a class initialised by decoder type?)

        if self.decoder == 'GCN':
            x1 = F.relu(self.gcn1(x_conv, self.A))
            x2 = F.relu(self.gcn2(x1,self.A))
            x3 = self.gcn3(x2,self.A)
            x3 = x3.squeeze()

        if self.decoder == 'GNN':
            #? what is the purpose of the gnn0 layer?
            # x0 = F.relu(self.gnn0(x_conv,self.A))
            x1 = F.relu(self.gnn1(x_conv,self.A))
            x2 = F.relu(self.gnn2(x1,self.A))
            x3 = self.gnn3(x2,self.A)
            x3 = x3.squeeze()
        if self.decoder == 'rGNN':
            x1 = F.relu(self.gc1(x_conv,self.adjs))
            # TODO: evaluate the influence of activating dropout layers
            # x1 = F.dropout(x1, self.dropout)
            x2 = F.relu(self.gc2(x1, self.adjs))
            x3 = F.relu(self.gc3(x2, self.adjs))
            # x3 = F.dropout(x2, self.dropout)
            x3 = x3.squeeze()

        if self.decoder == 'SAGE':
            x1 = F.relu(self.sage1(x_conv,self.A))
            x2 = F.relu(self.sage2(x1,self.A))
            x3 = F.relu(self.sage3(x2,self.A))
            x3 = x3.squeeze()

        if self.decoder == 'GIN':
            x3 = F.relu(self.gin(x_conv, self.A))
            x3 = x3.squeeze()

        if self.decoder == 'GAT':
            x1 = F.relu(self.gatconv1(x_conv,self.edge_index))
            x2 = F.relu(self.gatconv2(x1,self.edge_index))
            x3 = F.relu(self.gatconv3(x2,self.edge_index))
            x3 = x3.squeeze()

        if self.hw>0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1)
            z = self.highway(z)
            z = z.squeeze(2)
            x3 = x3 + z
        
        return x3
    

    def _set_A(self, A_new: np.ndarray):
        # divide A by the sum over axis=0 --> normalisation?
        A = A/np.sum(A, 0)
        A_new: np.ndarray = np.zeros((self.BATCH_SIZE, self.n_e, self.n_e), dtype=np.float32)
        for i in range(self.BATCH_SIZE):
            A_new[i,:,:] = A

        if self.use_cuda:
            self.A: torch.Tensor = torch.from_numpy(A_new).cuda()
        else:
            self.A: torch.Tensor = torch.from_numpy(A_new).cpu()

        self.adjs = [self.A]
        
        

    def _set_adjs(self, B: np.ndarray):
        # I believe they're setting this using A and A_new to save memory
        A = np.loadtxt(B)
        A = np.array(A, dtype=np.float32)
        # divide A by the sum over axis=1 --> why different than for A?
        A = A / np.sum(A, 1)
        A_new = np.zeros((self.BATCH_SIZE, self.n_e, self.n_e), dtype=np.float32)
        for i in range(self.BATCH_SIZE):
            A_new[i, :, :] = A
        
        if self.use_cuda:
            self.B = torch.from_numpy(A_new).cuda()
        else:
            self.B = torch.from_numpy(A_new).cpu()

        # I believe they're setting this using A and A_new to save memory
        A = np.ones((self.n_e, self.n_e),np.int8)
        # divide A by the sum over axis=1 --> why different than for A?
        A = A / np.sum(A, 1)
        A_new = np.zeros((self.BATCH_SIZE, self.n_e, self.n_e), dtype=np.float32)
        for i in range(self.BATCH_SIZE):
            A_new[i, :, :] = A
        self.C = torch.from_numpy(A_new).cuda()
        self.adjs = [self.A,self.B,self.C]
