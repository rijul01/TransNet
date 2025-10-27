import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from torch_geometric.utils import get_laplacian

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class graph(nn.Module):
    def __init__(self, iInd, jInd, nnodes, W, device, pos=None, faces=None):
        """
        Initialize graph structure for processing batched data.
        
        Args:
            iInd (torch.Tensor): Source node indices with shape (batch_size, num_edges)
            jInd (torch.Tensor): Target node indices with shape (batch_size, num_edges)
            nnodes (int): Number of nodes in the graph
            W (torch.Tensor): Edge weights with shape (batch_size, num_edges)
            device (torch.device): Device to place the tensors on
            pos (torch.Tensor, optional): Node positions 
            faces (torch.Tensor, optional): Face definitions for mesh structures
        """
        super(graph, self).__init__()
        self.device = device
        self.iInd = iInd.long().to(device)  # Source indices for each batch
        self.jInd = jInd.long().to(device)  # Target indices for each batch
        self.nnodes = nnodes
        self.W = W.to(device)
        self.pos = pos.to(device) if pos is not None else None
        self.faces = faces.to(device) if faces is not None else None
        
        # print(f"Graph initialized with:")
        # print(f"- Number of nodes: {nnodes}")
        # print(f"- Edge indices shape: {iInd.shape}")
        # print(f"- Edge weight shape: {W.shape}")
    
    
    def get_normalized_laplacian(self):
        # Ensure edge_weights are float and converted
        edge_weights = self.W.float()
    
        # Create sparse adjacency matrix with consistent types
        device = edge_weights.device
        adj = torch.zeros((self.nnodes, self.nnodes), 
                          dtype=torch.float32, 
                          device=device)
    
        # Convert indices and weights to appropriate types
        iInd = self.iInd.long()
        jInd = self.jInd.long()
    
        # Symmetric adjacency
        adj[iInd, jInd] = edge_weights
        adj[jInd, iInd] = edge_weights
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        
        # Create inverse degree matrix (with small epsilon to prevent division by zero)
        inv_degree = torch.pow(degree + 1e-10, -0.5)
        
        # Normalized Laplacian computation
        I = torch.eye(self.nnodes, dtype=torch.float32, device=device)
        D_inv_sqrt = torch.diag(inv_degree)
        L = I - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        return L
    
    def nodeGrad(self, x, W = None, replaceW = False):
        """
        Compute gradients between connected nodes for batched data.
        
        Args:
            x (torch.Tensor): Node features of shape (batch_size, channels, num_nodes)
            W (torch.Tensor, optional): Custom edge weights of shape (batch_size, num_edges)
            replaceW (bool): Whether to replace or multiply with existing weights
            
        Returns:
            torch.Tensor: Gradients of shape (batch_size, channels, num_edges)
        """
        if W is None:
            W = self.W
        else:
            if not replaceW:
                W = self.W * W
        
        batch_size, channels, num_nodes = x.shape
        
        # Compute gradients: Batch-wise
        g = torch.zeros(batch_size, channels, self.iInd.shape[1], device=self.device)
        for b in range(batch_size):
            src_features = x[b, :, self.iInd[b]]
            tgt_features = x[b, :, self.jInd[b]]
            g[b] = W[b].unsqueeze(0) * (tgt_features - src_features)
        
        return g
        # if len(W) == 0:
        #     W = self.W
        # else:
        #     if not replaceW:
        #         W = self.W * W
        # if W.shape[0] == x.shape[2]:
        #     # if its degree matrix
        #     g = W[self.iInd] * W[self.jInd] * (x[:, :, self.iInd] - x[:, :, self.jInd])
        # else:
        #     g = W * (x[:, :, self.iInd] - x[:, :, self.jInd])

        # return g

    def dirichletEnergy(self, x, W):
        g = (W[self.iInd] * x[:, :, self.iInd] - W[self.jInd] * x[:, :, self.jInd])
        return g

    def nodeAve(self, x, W = None):
        """
        Compute average features between connected nodes for batched data.
        
        Args:
            x (torch.Tensor): Node features of shape (batch_size, channels, num_nodes)
            W (torch.Tensor, optional): Custom edge weights of shape (batch_size, num_edges)
            
        Returns:
            torch.Tensor: Averaged features of shape (batch_size, channels, num_edges)
        """
        
        if len(W) == 0:
            W = self.W
        
        batch_size, channels, num_nodes = x.shape
        
        g = torch.zeros(batch_size, channels, self.iInd.shape[1], device=self.device)
        for b in range(batch_size):
            # get source and target feature of this batch
            src_features = x[b, :, self.iInd[b]]
            tgt_features = x[b, :, self.jInd[b]]
            g[b] = W[b].unsqueeze(0) * (src_features + tgt_features) / 2.0
        return g
        
        # if W.shape[0] == x.shape[2]:
        #     g = W[self.iInd] * W[self.jInd] * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        # else:
        #     g = W * (x[:, :, self.iInd] + x[:, :, self.jInd]) / 2.0
        # return g
        
    def neighborNode(self, x, W=[], replaceW=False):
        # x: [batch_size, hidden_dim, n_stations]
        # W: [batch_size, num_edges]
        if len(W) == 0:
            W = self.W
        else:
            if not replaceW:
                W = self.W * W
    
        # [batch_size, num_edges] -> [batch_size, 1, num_edges]
        batch_size = x.shape[0]
        hidden_dim = x.shape[1]
        
        x_dest = torch.stack([
            x[b, :, self.jInd[b]] 
            for b in range(batch_size)
        ])  # Should give [batch_size, hidden_dim, num_edges]
        
        W = W.view(batch_size, 1, -1).expand(-1, hidden_dim, -1)
        g = W * x_dest
        return g    
    
    
    # def neighborNode(self, x, W=[], replaceW=False):
    #     # x is of shape [num_nodes, channels]
    #     # print(x.shape, W.shape)
    #     # safgsd
    #     if len(W) == 0:
    #         W = self.W
    #     else:
    #         if not replaceW:
    #             W = self.W * W
    #             # W = softmax(W, self.iInd, num_nodes=x.shape[-1])
    #     if W.shape[0] == x.shape[2]:
    #         g = W[self.iInd] * (x[:, :, self.jInd])
    #         # g = W[self.iInd].t().unsqueeze(1) * (x[:, :, self.jInd])
    #     else:
    #         if x.shape[0] == 1:
    #             g = W * (x[:, :, self.jInd])
                
    #         else:
    #             g = W[self.iInd].t().unsqueeze(1) * (x[:, :, self.jInd])  # .transpose(-1, 0)
    #     # g is of shape [2*num_edges, channels]
    #     return g

    def neighborEdge(self, g, W=[], replaceW=False):
        ###  *-----*
        # g is of shape [channels, num_edges]
        if len(W) == 0:
            W = self.W
        else:
            if not replaceW:
                W = self.W * W
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=self.device)
        if W.shape[0] != g.shape[2]:
            x2.index_add_(2, self.iInd, W[self.iInd] * g)
        else:
            x2.index_add_(2, self.iInd, W * g)

        x = x2
        # x is of shape [num_nodes, channels]
        return x

    def edgeDiv(self, g, W = None):
        """
        Compute divergence of edge features back to nodes for batched data.
        
        Args:
            g (torch.Tensor): Edge features of shape (batch_size, channels, num_edges)
            W (torch.Tensor, optional): Custom edge weights of shape (batch_size, num_edges)
            
        Returns:
            torch.Tensor: Node features of shape (batch_size, channels, num_nodes)
        """
        if W is None:
            W = self.W
        
        batch_size, channels, num_edges = g.shape
        
        x = torch.zeros(batch_size, channels, self.nnodes, device=g.device, dtype = g.dtype)
        
        for b in range(batch_size):
            weighted_features = W[b].unsqueeze(0).to(g.dtype) * g[b]
            x[b].index_add_(1, self.iInd[b], weighted_features)
            x[b].index_add_(1, self.jInd[b], -weighted_features)
        
        return x
        # if len(W) == 0:
        #     W = self.W
        # if False:
        #     Wtmp = W.clone()
        #     _, indices = Wtmp.topk(100)
        #     gtmp = g.clone()
        #     gtmp[:, :, indices]
        #     rel_iInd = self.iInd[indices]
        #     rel_jInd = self.jInd[indices]
        #     nnodes = max(self.iInd[indices].unique().numel(), self.jInd[indices].unique().numel())
        #     xtmp = torch.zeros(g.shape[0], g.shape[1], nnodes, device=g.device)
        #     if W.shape[0] != g.shape[2]:
        #         xtmp.index_add_(2, self.iInd, W[self.iInd] * g)
        #         xtmp.index_add_(2, self.iInd, W[self.jInd] * g)
        #     else:
        #         xtmp.index_add_(2, self.iInd, Wtmp * g)
        #         xtmp.index_add_(2, self.jInd, -Wtmp * g)

        # x = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        # # z = torch.zeros(g.shape[0],g.shape[1],self.nnodes,device=g.device)
        # # for i in range(self.iInd.numel()):
        # #    x[:,:,self.iInd[i]]  += w*g[:,:,i]
        # # for j in range(self.jInd.numel()):
        # #    x[:,:,self.jInd[j]] -= w*g[:,:,j]
        # if W.shape[0] != g.shape[2]:
        #     x.index_add_(2, self.iInd, W[self.iInd] * g)
        #     x.index_add_(2, self.iInd, W[self.jInd] * g)
        # else:
        #     print(g.shape, W.shape)
        #     x.index_add_(2, self.iInd, W * g)
        #     x.index_add_(2, self.jInd, -W * g)

        # return x

    def edgeAve(self, g, method='max', W=[]):
        if len(W) == 0:
            W = self.W
        x1 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=self.device)
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=self.device)
        if W.shape[0] != g.shape[2]:
            x1.index_add_(2, self.iInd, W[self.iInd] * g)
            x2.index_add_(2, self.jInd, W[self.jInd] * g)
        else:
            x1.index_add_(2, self.iInd, W * g)
            x2.index_add_(2, self.jInd, W * g)
        if method == 'max':
            x = torch.max(x1, x2)
        elif method == 'ave':
            x = (x1 + x2) / 2
        return x

    def nodeLap(self, x):
        g = self.nodeGrad(x)
        d = self.edgeDiv(g)
        return d

    def edgeLength(self, x):
        g = self.nodeGrad(x)
        L = torch.sqrt(torch.pow(g, 2).sum(dim=1))
        return L

    def nodeProd(self, S):
        # SP = torch.bmm(S[:, :, self.iInd].transpose(2, 0).transpose(2, 1),
        #                S[:, :, self.jInd].transpose(2, 0)).transpose(2, 0)
        SP = S[:, :, self.iInd] * S[:, :, self.jInd]

        return SP