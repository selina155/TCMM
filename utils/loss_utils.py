import torch

def temporal_graph_sparsity(G: torch.Tensor):
    """
    Args:
    G: (lag+1, num_nodes, num_nodes) or (batch, lag+1, num_nodes, num_nodes)

    Returns:
    Square of Frobenius norm of G (batch, )
    """
    return torch.sum(torch.square(G))

def hierarchical_temporal_graph_sparsity(G: torch.Tensor):
    """
    Args:
    G: (lag+1, num_nodes, num_nodes) or (batch, lag+1, num_nodes, num_nodes)
    lam: Regularization coefficient

    Returns:
    Regularization term with hierarchical sparsity (batch, )
    """
    if G.ndimension() == 4:
        G=G.flip([1])
        num_graphs, lags, num_nodes, _ = G.shape
        reg_term = 0.0
        for i in range(lags):
            reg_term += torch.sum(torch.square(G[:, :(i+1), :, :]))
        return reg_term

    elif G.ndimension() == 3:
        G=G.flip([0])
        lags, num_nodes, _ = G.shape
        reg_term=0.0
        for i in range(lags):
            reg_term += torch.sum(torch.square(G[:(i+1),:,:]))
        return reg_term
    else:
        raise ValueError("Invalid shape for G. Expected 3 or 4 dimensions.")

def l1_sparsity(G: torch.Tensor):
    return torch.sum(torch.abs(G))


def dag_penalty_notears(G: torch.Tensor):
    """
    Implements the DAGness penalty from
    "DAGs with NO TEARS: Continuous Optimization for Structure Learning"

    Args:
    G: (num_nodes, num_nodes) or (num_graphs, num_nodes,  num_nodes)
    """
    num_nodes = G.shape[-1]

    if len(G.shape) == 2:
        trace_term = torch.trace(torch.matrix_exp(G))
        return trace_term - num_nodes
    elif len(G.shape) == 3:
        trace_term = torch.einsum("ijj->i", torch.matrix_exp(G))
        return torch.sum(trace_term - num_nodes)
    assert False, "DAG Penalty received illegal shape"


def dag_penalty_notears_sq(W: torch.Tensor):
    num_nodes = W.shape[-1]

    if len(W.shape) == 2:
        trace_term = torch.trace(torch.matrix_exp(W * W))
        return trace_term - num_nodes
    elif len(W.shape) == 3:
        trace_term = torch.einsum("ijj->i", torch.matrix_exp(W * W))
        return torch.sum(trace_term) - W.shape[0] * num_nodes
    assert False, "DAG Penalty received illegal shape"
