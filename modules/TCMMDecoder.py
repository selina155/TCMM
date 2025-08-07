import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.modules.MultiEmbedding import MultiEmbedding,Embedding

class SpatioTemporalGraphAttentionLayer(pl.LightningModule):
    """
    X_input:(batch,num_graphs, lag + 1, num_nodes, data_dim)
    A: (batch,num_graphs, lag+1, num_nodes, num_nodes)
    output: (batch,num_graphs, num_nodes, data_dim)
    """
    def __init__(self,
                 data_dim:int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim:int,
                 dropout=0
                 ):
        super(SpatioTemporalGraphAttentionLayer, self).__init__()
        self.embedding_dim=embedding_dim
        self.dropout = dropout
        self.in_features = data_dim
        self.num_graphs=num_graphs
        self.num_nodes=num_nodes
        self.lag=lag
        self.layernorm=nn.LayerNorm(self.in_features)
        self.input_mapping=self._feature_mapping(input_dim=self.in_features+self.embedding_dim)
        self.t_feat_mapping=self._feature_mapping(input_dim=2*self.embedding_dim)
        self.out_mapping=self._feature_mapping(input_dim=self.in_features+self.embedding_dim,use_activation=True)
        # single graph setting
        if self.num_graphs is None:
            self.embeddings = Embedding(num_nodes=num_nodes,lag=self.lag,embedding_dim=self.embedding_dim)
        # mixture graph setting
        else:
            self.embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                                lag=self.lag,
                                                num_graphs=self.num_graphs,
                                                embedding_dim=self.embedding_dim)
        self.a=nn.Linear(2*self.in_features,1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h,adj):
        batch, num_graphs, L, num_nodes, data_dim = h.shape
        E=self.embeddings.get_embeddings()
        E = E.unsqueeze(0).expand((h.shape[0], -1, -1, -1, -1))
        #intrinsic embedding at time t
        E_inst=E[:,:,-1,:,:].unsqueeze(2).expand((-1,-1,L-1,-1,-1))
        Wh = self.input_mapping(torch.cat((h,E),dim=-1))
        #proxy node feature generation, historical values of the current node are concatenated with E_inst,
        previous=self.input_mapping(torch.cat((h[:,:,:-1],E_inst),dim=-1))
        # E_inst is concatenated with itself at t to learn instantaneous attention
        t=self.t_feat_mapping(torch.cat((E[:,:,-1,:,:],E[:,:,-1,:,:]),dim=-1))
        t_proxy=torch.cat((previous,t.unsqueeze(2)),dim=2)
        #attention compution
        a_input = self._prepare_attentional_mechanism_input(Wh,t_proxy)
        e = self.leakyrelu(self.a(a_input))#(batch,K,T,D,D)
        attention =  e.squeeze(-1)
        attention = attention.permute(0, 1, 3, 2, 4).reshape(batch, num_graphs, num_nodes, -1)
        attention = F.softmax(attention, dim=-1)
        adj = adj.permute(0, 1, 3,4,2).reshape(batch, num_graphs, num_nodes, -1)
        # GraphSoftmax
        causal_att=self._graphsoftmax(attention,adj)
        #Causal parent information Aggregate
        X_sum = torch.einsum("bnij,bnjd->bnid", causal_att, Wh.reshape(batch,num_graphs,L*num_nodes,self.in_features))
        h=self.out_mapping(torch.cat((X_sum,E[:,:,-1,:,:]),dim=-1))
        return h
    def _feature_mapping(self, input_dim, hidden_dim=128, output_dim=1, use_activation=False):
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_activation:
            layers.append(nn.LeakyReLU())  
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    def _graphsoftmax(self,attention,A):
        causal_att=attention*A
        causal_att=causal_att.float()
        mask=(causal_att != 0).float()
        softmax_causal_att=F.softmax(causal_att+(1.0-mask)*-1e9,dim=-1)
        masked_att=softmax_causal_att*mask
        masked_att = F.dropout(masked_att, self.dropout, training=self.training)
        return masked_att
    def _prepare_attentional_mechanism_input(self,Wh,t_proxy):
        N,K,T,D,C=Wh.shape
        Wh = Wh.reshape(N * K, T, D, C)
        t_proxy= t_proxy.reshape(N * K, T, D, C)
        Wh_neighbor= Wh.unsqueeze(2).expand(-1, -1, D, -1, -1)
        Wh_current = t_proxy.view(N * K, T, 1, D, C).expand(-1, -1, D, -1, -1)
        Wh_current = Wh_current.permute(0, 1, 3, 2, 4)
        a_input = torch.cat([Wh_neighbor, Wh_current], dim=-1)
        return a_input.view(N, K,T,D, D, 2* C)


class SpatiotemporalGAT(nn.Module):
    """
    X_input:(batch,num_graphs, lag + 1, num_nodes, data_dim)
    A: (batch,num_graphs, lag+1, num_nodes, num_nodes)
    output: (batch,num_graphs, num_nodes, data_dim)
    Args:
    num_nodes: the number of the nodes
    num_graphs: the number of the causal graphs
    num_layers: the number of spatiotemporal graph attention layers
    num_heads: the number of attention heads
    embedding_dim: dimension of the learnable embedding
    """
    def __init__(self, data_dim:int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int=None,
                 num_layers:int=1,
                 num_heads: int=1,
                 embedding_dim:int=32,
                  dropout=0):
        super(SpatiotemporalGAT, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_dim=data_dim
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.lag=lag
        self.num_graphs=num_graphs
        self.num_layers=num_layers
        self.input_dim=self.data_dim
        self.layers=nn.ModuleList()
        self.layer_norms=nn.ModuleList()
        for _ in range(num_layers):
            attention_heads=nn.ModuleList([
                SpatioTemporalGraphAttentionLayer(
                    data_dim=self.data_dim,
                    lag=self.lag,num_nodes=self.num_nodes,
                    num_graphs=self.num_graphs,
                    embedding_dim=embedding_dim,
                    dropout=dropout) for _ in range(num_heads)
                ])
            self.layers.append(attention_heads)
            self.layer_norms.append(nn.LayerNorm(self.input_dim))
        self.BN=nn.BatchNorm2d(self.data_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x,A):
        x=self.BN(x.permute(0,3,2,1)).permute(0,3,2,1)
        if self.num_graphs is None:
            adj=A.unsqueeze(1)
            h = x.unsqueeze(1)
        else:
            adj = A.unsqueeze(0).expand((x.shape[0], -1, -1, -1, -1))
            h = x.unsqueeze(1).expand(
                (-1, self.num_graphs, -1, -1, -1))      
        adj = adj.flip([2])
        for layer_idx in range(self.num_layers):
            residual=h[:,:,-2] #pervious value of predicting node
            X=[]
            for att in self.layers[layer_idx]:
                x_head=att(h,adj)
                x_head=self.layer_norms[layer_idx](x_head+residual)
                X.append(x_head)
            h_pred=torch.mean(torch.cat(X,dim=-1),dim=-1).unsqueeze(-1)
            h_pred = F.dropout(h_pred, self.dropout, training=self.training)
        if self.num_graphs is None:
            return h_pred.squeeze(1)
        else:
            return h_pred
             
