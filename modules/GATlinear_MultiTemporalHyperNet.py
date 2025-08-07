from typing import Dict, Tuple
import lightning.pytorch as pl
from torch import nn
import torch
from src.modules.MultiEmbedding import MultiEmbedding
from src.utils.torch_utils import generate_fully_connected
import torch.nn.functional as F


class MultiTemporalHyperNet(pl.LightningModule):

    def __init__(self,
                 order: str,
                 lag: int,
                 data_dim: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim: int = None,
                 skip_connection: bool = False,
                 num_bins: int = 8,
                 dropout_p: float = 0.0
                 ):

        super().__init__()

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = num_nodes * data_dim

        self.data_dim = data_dim
        self.lag = lag
        self.order = order
        self.num_bins = num_bins
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.dropout_p = dropout_p

        if self.order == "quadratic":
            self.param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
            ]  # this is for quadratic order conditional spline flow
        elif self.order == "linear":
            self.param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
                self.num_bins,
            ]  # this is for linear order conditional spline flow

        self.total_param = sum(self.param_dim)
        input_dim = 2 * self.embedding_dim

        self.nn_size = max(4 * num_nodes, self.embedding_dim, 64)

        self.th_embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                            lag=self.lag,
                                            num_graphs=self.num_graphs,
                                            embedding_dim=self.embedding_dim)
        self.hidden_dim1=128
        self.hidden_dim2=128
        self.W1=nn.Linear(self.data_dim+self.embedding_dim,self.hidden_dim1)
        self.W2=nn.Linear(self.hidden_dim1,self.data_dim)
        self.W3 = nn.Linear(self.data_dim + self.embedding_dim, self.hidden_dim2)
        self.W4 = nn.Linear(self.hidden_dim2, self.total_param)
        self.leakyrelu = nn.LeakyReLU()
        self.a=nn.Linear(2*self.data_dim,1)

    def forward(self, X: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            X: A dict consisting of two keys, "A" is the adjacency matrix with shape [batch, lag+1, num_nodes, num_nodes]
            and "X" is the history data with shape (batch, lag, num_nodes, data_dim).

        Returns:
            A tuple of parameters with shape [N_batch, num_cts_node*param_dim_each].
                The length of tuple is len(self.param_dim),
        """

        # assert "A" in X and "X" in X and len(
        # X) == 2, "The key for input can only contain two keys, 'A', 'X'."

        A = X["A"]
        X_in = X["X"]
        E = self.th_embeddings.get_embeddings()
        # X["embeddings"]

        batch, lag, num_nodes, _ = X_in.shape

        # reshape X to the correct shape
        A = A.unsqueeze(0).expand((batch, -1, -1, -1, -1))
        E = E.unsqueeze(0).expand((batch, -1, -1, -1, -1))
        X_in = X_in.unsqueeze(1).expand((-1, self.num_graphs, -1, -1, -1))

        # ensure we have the correct shape
        assert (A.shape[0] == batch and A.shape[1] == self.num_graphs and
                A.shape[2] == lag+1 and A.shape[3] == num_nodes and
                A.shape[4] == num_nodes)
        assert (E.shape[0] == batch and E.shape[1] == self.num_graphs
                and E.shape[2] == lag+1 and E.shape[3] == num_nodes
                and E.shape[4] == self.embedding_dim)
        # shape [batch_size, num_graphs, lag, num_nodes, embedding_size]
        E_lag = E[:, :, :-1, :, :]
        A=A[:, :, 1:].flip([2])
        #A=A.flip([2])
        # reshape X to the correct shape
        X_in = torch.cat((X_in, E_lag), dim=-1)
        Wh=self.W2(self.W1(X_in))
        #a_input=self._prepare_attentional_mechanism_input(Wh)
        #e = self.leakyrelu(self.a(a_input))  # (N, num,V, V, T)
        #attention = e.squeeze(-1)
        #attention = attention.permute(0, 1, 3, 2, 4).reshape(batch, self.num_graphs, num_nodes, -1)
        #attention = F.softmax(attention, dim=-1)
        #adj = A.permute(0, 1, 3, 4, 2).reshape(batch, self.num_graphs, num_nodes, -1)
        #Wh1=torch.matmul(attention,Wh)
        X_sum = torch.einsum("bnlij,bnljd->bnid", A,Wh)
        h = self.W4(self.W3(torch.cat((X_sum, E[:, :, -1, :, :]), dim=-1)))
        params = h.reshape(batch * self.num_graphs, num_nodes, -1)

        param_list = torch.split(params, self.param_dim, dim=-1)
        # a list of tensor with shape [batch*num_graphs, num_nodes*each_param]
        return tuple(
            param.reshape([-1, num_nodes * param.shape[-1]]) for param in param_list)

