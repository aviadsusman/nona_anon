import torch
import torch.nn as nn

def sim_matrix(x, x_n, similarity):
    if similarity == 'l2':
        sim = - torch.cdist(x, x_n, p=2)
    elif similarity == 'l1':
        sim = - torch.cdist(x, x_n, p=1)
    elif similarity == 'dot':
        sim = x @ torch.t(x_n) / x.shape[1]
    elif similarity == 'cos':
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        x_n_norm = nn.functional.normalize(x_n, p=2, dim=1)
        sim = x_norm @ torch.t(x_n_norm)
    
    return sim

def norm_sim(sim):
    sim_min = sim.min(dim=1, keepdim=True)[0]
    sim_max = sim.max(dim=1, keepdim=True)[0]
    sim_norm = (sim - sim_min) / (sim_max - sim_min)
    return sim_norm

# Mask functions
def S_1(x, a, b, t, eps=1e-12):
    num = torch.abs(x - a).pow(1 / t).clamp(eps)
    denom = num + torch.abs(b - x).pow(1 / t).clamp(eps)

    mask = num / denom
    mask[x < a] = 0
    mask[x > b] = 1
    
    return mask.log()

def S_2(x, a, b, t, eps=1e-12):
    mask = (t / (1-t)) * (t - x) * (x / b).log()
    mask[x > b] = 1

    return mask

class SoftStep(nn.Module):
    def __init__(self, step_fn=2, dims=None, eps=1e-12):
        super().__init__()
        if step_fn == 1:
            self.step_fn = S_1
        elif step_fn == 2:
            self.step_fn = S_2

        self.eps = eps
        self.dims = dims

        if dims is None:
            # Uniform mask: shared learnable parameters
            self.params = nn.Parameter(torch.randn(3))
            self.pointwise = False
        else:
            # Pointwise mask: MLP that outputs [a, b, t] per input
            self.pointwise = True
            self.params = self.build_mlp(dims)

    def build_mlp(self, dims):
        if isinstance(dims, int):
            dims = [dims]
        layers = []
        if len(dims) == 1:
            layers += [nn.Linear(dims[0], 3), nn.Sigmoid()]
        else:
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.Tanh()]
            layers += [nn.Linear(dims[-1], 3), nn.Sigmoid()]
        return nn.Sequential(*layers)

    def forward(self, x, x_n, similarity):
        # Compute similarity
        sim = sim_matrix(x, x_n, similarity)
        sim_norm = norm_sim(sim)

        # Get parameters
        if self.pointwise:
            out = self.params(x)  # MLP(x) → [a, b, t]
            a, b, t = [col.unsqueeze(-1).clamp(self.eps, 1 - self.eps) for col in out.T]
        else:
            a, b, t = torch.sigmoid(self.params).clamp(self.eps, 1 - self.eps)

        # Adjust a to ensure at least one neighbor
        if torch.equal(x, x_n):
            top_sims = (sim_norm - sim_norm.diag().diag()).max(dim=1)[0]
        else:
            top_sims = sim_norm.max(dim=1)[0]
        a = torch.minimum(a, top_sims.unsqueeze(-1)) - self.eps

        # Compute softmask
        mask = self.step_fn(sim_norm, a, b, t)
        return sim + mask