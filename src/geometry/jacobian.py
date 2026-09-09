"""Analytic Jacobian of the shPLRNN. Exact, no autodiff, piecewise constant."""
import torch


def jacobian(s, model):
    """d(s_{t+1})/d(s_t) at s. Valid anywhere off a switching hyperplane (measure zero)."""
    D = model.gate(s)                                   # (H,)
    return torch.diag(model.A) + model.W1 @ torch.diag(D) @ model.W2


def batched_jacobian(S, model):
    """(N,d) -> (N,d,d)."""
    D = model.gate(S)                                   # (N,H)
    return torch.diag(model.A).expand(S.shape[0], -1, -1) + \
        torch.einsum("dh,nh,he->nde", model.W1, D, model.W2)


def on_hyperplane(s, model, tol=1e-6):
    """True if s sits within tol of any switching hyperplane -- the Jacobian is undefined
    there and FTLE values at such points should be treated with suspicion."""
    return bool((( s @ model.W2.T + model.h2).abs() < tol).any())
