import lightning.pytorch as pl
from torch import nn
import torch


class MultiEmbedding(pl.LightningModule):

    def __init__(
        self,
        num_nodes: int,
        lag: int,
        num_graphs: int,
        embedding_dim: int
    ):

        super().__init__()
        self.lag = lag
        # Assertion lag > 0
        assert lag > 0
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.embedding_dim = embedding_dim
        self.graph_emb=nn.Parameter((torch.randn(self.num_graphs,1,1,self.embedding_dim,device=self.device)*0.01),requires_grad=True)
        self.node_emb=nn.Parameter((torch.randn(1,1,self.num_nodes,self.embedding_dim,device=self.device)*0.01),requires_grad=True)

        self.lag_embeddings = nn.Parameter((
            torch.randn(1, self.lag, 1,self.embedding_dim, device=self.device) * 0.01), requires_grad=True)

        self.inst_embeddings = nn.Parameter((
        torch.randn(1, 1, 1,self.embedding_dim, device=self.device) * 0.01), requires_grad=True)

    def turn_off_inst_grad(self):
        self.inst_embeddings.requires_grad_(False)

    def turn_on_inst_grad(self):
        self.inst_embeddings.requires_grad_(True)

    def get_embeddings(self):
        temp_emb=torch.cat((self.lag_embeddings,self.inst_embeddings), dim=1)
        return(self.graph_emb+self.node_emb+temp_emb)

class Embedding(pl.LightningModule):
    def __init__(self,
                 num_nodes: int,
                 lag: int,
                 embedding_dim: int):
        super().__init__()
        self.lag=lag
        assert lag > 0
        self.num_nodes=num_nodes
        self.embedding_dim=embedding_dim
        self.node_emb=nn.Parameter((torch.randn(1,self.num_nodes,self.embedding_dim,device=self.device)*0.01),requires_grad=True)
        self.lag_emb=nn.Parameter((torch.randn(self.lag,1,self.embedding_dim,device=self.device)*0.01),requires_grad=True)
        self.inst_emb=nn.Parameter((torch.randn(1,1,self.embedding_dim,device=self.device)*0.01),requires_grad=True)
    def turn_off_inst_grad(self):
        self.inst_emb.requires_grad_(False)

    def turn_on_inst_grad(self):
        self.inst_emb.requires_grad_(True)

    def get_embeddings(self):
        temp_emb=torch.cat((self.lag_emb,self.inst_emb), dim=0)
        return(self.node_emb+temp_emb)
