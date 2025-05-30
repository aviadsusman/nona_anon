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
    mask = t / (1-t) * (b - x) * (x / b).log()
    mask = torch.where(x <= b, mask, 0)

    return mask

class SoftStep(nn.Module):
    def __init__(self, step_fn=2, mask_type='pointwise', eps=1e-12, dims=25):
        super().__init__()
        self.eps = eps
        if step_fn == 1:
            self.step_fn = S_1
        elif step_fn == 2:
            self.step_fn = S_2

        self.mask_type = mask_type
        self.dims = dims
        # Global m
        # ask: shared learnable parameters
        if self.mask_type == 'pointwise':
            self.pointwise = True
            self.params = nn.Linear(dims, 3)

        elif self.mask_type == 'global':
            self.params = nn.Parameter(torch.randn(3))

    def forward(self, x, x_n, similarity):
        # Compute similarity
        sim = sim_matrix(x, x_n, similarity)
        sim_norm = norm_sim(sim)

        # Get parameters
        if self.mask_type=='pointwise':
            out = torch.sigmoid(self.params(x)).clamp(self.eps, 1 - self.eps)
            a, b, t = [col.unsqueeze(-1) for col in out.T]
        else:
            a, b, t = torch.sigmoid(self.params).clamp(self.eps, 1 - self.eps)
        
        if self.step_fn == S_1:
        # Adjust a to ensure at least one neighbor
            if torch.equal(x, x_n):
                top_sims = (sim_norm - sim_norm.diag().diag()).max(dim=1)[0]
            else:
                top_sims = sim_norm.max(dim=1)[0]
            a = torch.minimum(a, top_sims.unsqueeze(-1)) - self.eps
            b = a + b * (1 - a)

        # Compute softmask
        mask = self.step_fn(sim_norm, a, b, t)
        return sim + mask