"""A simplicial neural network over the sphere's complex.

Edge branch (1-cochains, the wind as flow along edges): simplicial convolutions that mix each edge's signal with its
lower neighbours through L1_down = B1^T B1 (flows meeting at shared vertices: convergence and divergence) and its upper
neighbours through L1_up = B2 B2^T (flows around shared triangles: circulation). Filters are polynomials in each
Laplacian (Ebli et al. 2020; Yang et al. 2022), so each layer sees two hops of the complex.

Lift to vertices through the boundary map: B1 X (signed: net inflow/outflow, a discrete divergence of the learned
flows) and |B1| X (unsigned: local activity), averaged over each vertex's degree.

Vertex branch: the lifted features plus vertex inputs, through graph convolutions with the normalised L0 = B1 B1^T,
to a logit per vertex.

Laplacians are scaled by their largest eigenvalue so filter coefficients stay well conditioned.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from torch import nn

from .complex import Complex


def _torch_sparse(m: sp.spmatrix, device) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape).coalesce().to(device)


def _scaled(m: sp.spmatrix) -> sp.spmatrix:
    lam = sla.eigsh(m.astype(float), k=1, which="LM", return_eigenvectors=False)[0]
    return m / lam


class Operators:
    def __init__(self, cx: Complex, device):
        B1 = cx.B1.astype(float)
        deg = np.asarray(abs(B1).sum(1)).ravel()
        Dinv = sp.diags(1 / deg)
        self.Ld = _torch_sparse(_scaled(cx.L1_down), device)
        self.Lu = _torch_sparse(_scaled(cx.L1_up), device)
        self.L0 = _torch_sparse(_scaled(cx.L0), device)
        self.B1s = _torch_sparse(Dinv @ B1, device)        # signed lift, degree-normalised
        self.B1a = _torch_sparse(Dinv @ abs(B1), device)   # unsigned lift


class SimplicialConv(nn.Module):
    """X' = X W0 + sum_{k=1..K} (Ld^k X Wd_k + Lu^k X Wu_k)."""

    def __init__(self, cin: int, cout: int, K: int = 2):
        super().__init__()
        self.K = K
        self.w0 = nn.Linear(cin, cout)
        self.wd = nn.ModuleList(nn.Linear(cin, cout, bias=False) for _ in range(K))
        self.wu = nn.ModuleList(nn.Linear(cin, cout, bias=False) for _ in range(K))

    def forward(self, x, ops: Operators):
        out = self.w0(x)
        xd = xu = x
        for k in range(self.K):
            xd = torch.sparse.mm(ops.Ld, xd)
            xu = torch.sparse.mm(ops.Lu, xu)
            out = out + self.wd[k](xd) + self.wu[k](xu)
        return out


class GraphConv(nn.Module):
    def __init__(self, cin: int, cout: int, K: int = 2):
        super().__init__()
        self.ws = nn.ModuleList(nn.Linear(cin, cout, bias=(k == 0)) for k in range(K + 1))

    def forward(self, x, ops: Operators):
        out, h = self.ws[0](x), x
        for w in self.ws[1:]:
            h = torch.sparse.mm(ops.L0, h)
            out = out + w(h)
        return out


class TopoNet(nn.Module):
    def __init__(self, edge_in: int, vertex_in: int, hidden: int = 32):
        super().__init__()
        self.e1 = SimplicialConv(edge_in, hidden)
        self.e2 = SimplicialConv(hidden, hidden)
        self.v1 = GraphConv(2 * hidden + vertex_in, hidden)
        self.v2 = GraphConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def forward(self, x_e, x_v, ops: Operators):
        h = self.act(self.e1(x_e, ops))
        h = self.act(self.e2(h, ops))
        lifted = torch.cat([torch.sparse.mm(ops.B1s, h), torch.sparse.mm(ops.B1a, h)], 1)
        z = self.act(self.v1(torch.cat([lifted, x_v], 1), ops))
        z = self.act(self.v2(z, ops))
        return self.head(z).squeeze(-1)


class VertexOnly(nn.Module):
    """Ablation: same vertex network, no simplicial edge branch (wind enters only as vertex speed)."""

    def __init__(self, vertex_in: int, hidden: int = 32):
        super().__init__()
        self.v1 = GraphConv(vertex_in + 1, hidden)
        self.v2 = GraphConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def forward(self, x_e, x_v, ops: Operators):
        speed = torch.sparse.mm(ops.B1a, x_e[:, 1:2])
        z = self.act(self.v1(torch.cat([x_v, speed], 1), ops))
        z = self.act(self.v2(z, ops))
        return self.head(z).squeeze(-1)
