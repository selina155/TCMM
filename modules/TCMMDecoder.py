import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.modules.MultiEmbedding import MultiEmbedding,Embedding
from torch_geometric.nn import MessagePassing

class GraphAttentionLayer(pl.LightningModule):
    """
    X_input:(batch,num_graphs, lag + 1, num_nodes, data_dim)
    A: (batch,num_graphs, lag+1, num_nodes, num_nodes)
    output: (batch,num_graphs, num_nodes, data_dim)0
    """
    def __init__(self,
                 data_dim:int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim:int,
                 dropout=0
                 ):
        super(GraphAttentionLayer, self).__init__()
        self.embedding_dim=embedding_dim
        self.out_features=1
        self.dropout = dropout
        self.in_features = data_dim
        self.hidden_dim1=128
        self.num_graphs=num_graphs
        self.num_nodes=num_nodes
        self.lag=lag
        self.hidden_dim2=128
        self.layernorm=nn.LayerNorm(self.in_features)

        self.W1 = nn.Linear(self.in_features+self.embedding_dim, self.hidden_dim1)
        self.W2=nn.Linear(self.hidden_dim1,self.out_features)
        self.W5 = nn.Linear(self.in_features+self.embedding_dim, self.hidden_dim1)
        self.W6=nn.Linear(self.hidden_dim1,self.out_features)
        self.W55 = nn.Linear(2*self.embedding_dim, self.hidden_dim1)
        self.W66=nn.Linear(self.hidden_dim1,self.out_features)

        self.W3 = nn.Linear(self.out_features+self.embedding_dim, self.hidden_dim2)

        self.W4=nn.Linear(self.hidden_dim2,self.out_features)


        if self.num_graphs is None:#单图设置
            self.embeddings = Embedding(num_nodes=num_nodes,lag=self.lag,embedding_dim=self.embedding_dim)
        else:#混合图设置
            self.embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                                lag=self.lag,
                                                num_graphs=self.num_graphs,
                                                embedding_dim=self.embedding_dim)
        self.a=nn.Linear(2*self.out_features,1)
        self.leakyrelu = nn.LeakyReLU()


    def forward(self, h,adj):
        batch, num_graphs, L, num_nodes, data_dim = h.shape
        E=self.embeddings.get_embeddings()
        E = E.unsqueeze(0).expand((h.shape[0], -1, -1, -1, -1))
        E_inst=E[:,:,-1,:,:].unsqueeze(2).expand((-1,-1,L-1,-1,-1))
        Wh = self.W2(self.W1(torch.cat((h,E),dim=-1)))
        previous=self.W6(self.W5(torch.cat((h[:,:,:-1],E_inst),dim=-1)))
        t=self.W66(self.W55((torch.cat((E[:,:,-1,:,:],E[:,:,-1,:,:]),dim=-1))))
        t_agent=torch.cat((previous,t.unsqueeze(2)),dim=2)
           
        a_input = self._prepare_attentional_mechanism_input(Wh,t_agent)  
        
        e = self.leakyrelu(self.a(a_input))#(N, num,V, V, T)
        attention =  e.squeeze(-1)
        attention = attention.permute(0, 1, 3, 2, 4).reshape(batch, num_graphs, num_nodes, -1)
        attention = F.softmax(attention, dim=-1)

        adj = adj.permute(0, 1, 3,4,2).reshape(batch, num_graphs, num_nodes, -1)
        causal_att=attention*adj#causal mask
        #GraphSoftmax
        causal_att=causal_att.float()
        mask=(causal_att != 0).float()
        softmax_causal_att=F.softmax(causal_att+(1.0-mask)*-1e9,dim=-1)
        masked_att=softmax_causal_att*mask
        masked_att = F.dropout(masked_att, self.dropout, training=self.training)
        #Aggregate
        X_sum = torch.einsum("bnij,bnjd->bnid", masked_att, Wh.reshape(batch,num_graphs,L*num_nodes,self.out_features))
        h=self.W4(self.W3(torch.cat((X_sum,E[:,:,-1,:,:]),dim=-1)))
        h=self.leakyrelu(h)
        return h

    def _prepare_attentional_mechanism_input(self,Wh,t_agent): 
    
        N,num,T,V,C=Wh.shape
        Wh = Wh.reshape(N * num, T, V, C)
        t_agent= t_agent.reshape(N * num, T, V, C)

        Wh_repeated = Wh.unsqueeze(2).expand(-1, -1, V, -1, -1)  

        Wh_last=t_agent
        Wh_tiled = Wh_last.view(N * num, T, 1, V, C).expand(-1, -1, V, -1, -1)
        Wh_tiled = Wh_tiled.permute(0, 1, 3, 2, 4)  

        a_input = torch.cat([Wh_repeated, Wh_tiled], dim=-1)
        return a_input.view(N, num,T,V, V, 2* C)
        #


class SpatiotemporalGAT(nn.Module):
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
        print("num_graphs:",self.num_graphs)
        self.num_layers=num_layers
        self.input_dim=self.data_dim
        self.hidden_dim=128
        self.layers=nn.ModuleList()
        self.layer_norms=nn.ModuleList()
        for _ in range(num_layers):
            attention_heads=nn.ModuleList([
                GraphAttentionLayer(
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
            residual=h[:,:,-2]
            X=[]
            for att in self.layers[layer_idx]:
                x_head=att(h,adj)
                x_head=self.layer_norms[layer_idx](x_head+residual)
                X.append(x_head)
            h=torch.mean(torch.cat(X,dim=-1),dim=-1).unsqueeze(-1)
            h = F.dropout(h, self.dropout, training=self.training)
        if self.num_graphs is None:
            return h.squeeze(1)
        else:
            return h
             
