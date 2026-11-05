#!/usr/bin/env python3
"""
coastal_physics.py  --  linear rotating shallow-water physics for a coastal
f-plane domain (New York Bight / harbour entrance).

Separate from the verified equatorial module (physics.py). Equations:

    u_t - f v + g eta_x + r u = 0
    v_t + f u + g eta_y + r v = 0
    eta_t + d/dx(H u) + d/dy(H v) = 0

with H(x,y)>0 still-water depth, f = 2 Omega sin(phi0) a constant (f-plane at a
reference latitude), r>=0 an optional linear drag. Coordinates are local SI:
x eastward metres, y northward metres, t seconds; u,v in m/s; eta in m.

The continuity flux divergence is d/dx(Hu)+d/dy(Hv) = Hx u + H u_x + Hy v + H v_y,
so H and its gradients (Hx,Hy) must be supplied at the collocation points. For
the synthetic constant-H test Hx=Hy=0.

NOTE (observation model): HF radar measures NEAR-SURFACE velocity (~upper 0.5 m),
while these equations describe DEPTH-AVERAGED velocity. They are not the same
quantity; for a short tide-dominated interval near-surface velocity is used as a
proxy for the barotropic flow. See README.
"""

import math
import numpy as np
import torch
import torch.nn as nn

OMEGA = 7.292115e-5     # rad/s
G = 9.81               # m/s^2


def coriolis(phi0_deg):
    return 2.0 * OMEGA * math.sin(math.radians(phi0_deg))


def grad(q, X):
    if not q.requires_grad:
        return torch.zeros_like(X)
    g = torch.autograd.grad(q, X, grad_outputs=torch.ones_like(q),
                            create_graph=True, retain_graph=True,
                            allow_unused=True)[0]
    return torch.zeros_like(X) if g is None else g


# -------------------------------------------------------------------------
# Exact Poincare (inertia-gravity) plane wave on an f-plane, constant H.
#   eta = A cos(theta),  theta = k x + l y - omega t,
#   omega^2 = f^2 + g H (k^2 + l^2)
# Polarisation (derived from the linear equations):
#   u = gA/(w^2-f^2) [ k w cos - l f sin ]
#   v = gA/(w^2-f^2) [ l w cos + k f sin ]
# Used only to VERIFY the residual operator; r=0 for this exact solution.
# -------------------------------------------------------------------------
def poincare_solution(X, A, k, l, H, f):
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    w = math.sqrt(f * f + G * H * (k * k + l * l))
    theta = k * x + l * y - w * t
    ct, st = torch.cos(theta), torch.sin(theta)
    denom = (w * w - f * f)
    eta = A * ct
    u = G * A / denom * (k * w * ct - l * f * st)
    v = G * A / denom * (l * w * ct + k * f * st)
    return u, v, eta, w


def coastal_residuals(model, X, H, Hx, Hy, f, r=0.0):
    """
    PDE residuals at points X (requires_grad set internally). H, Hx, Hy are
    tensors (N,1) giving depth and its gradients at X. r may be a float or a
    trainable tensor. model(X) -> (u,v,eta).
    """
    X = X.clone().detach().requires_grad_(True)
    u, v, eta = model(X)
    gu, gv, ge = grad(u, X), grad(v, X), grad(eta, X)
    ux, uy, ut = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    vx, vy, vt = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    ex, ey, et = ge[:, 0:1], ge[:, 1:2], ge[:, 2:3]

    Ru = ut - f * v + G * ex + r * u
    Rv = vt + f * u + G * ey + r * v
    Reta = et + (Hx * u + H * ux) + (Hy * v + H * vy)
    return Ru, Rv, Reta


def residuals_from_fields(u, v, eta, X, H, Hx, Hy, f, r=0.0):
    """Same operator but for externally supplied analytic fields (for tests)."""
    gu, gv, ge = grad(u, X), grad(v, X), grad(eta, X)
    ux, ut = gu[:, 0:1], gu[:, 2:3]
    vy, vt = gv[:, 1:2], gv[:, 2:3]
    ex, ey, et = ge[:, 0:1], ge[:, 1:2], ge[:, 2:3]
    Ru = ut - f * v + G * ex + r * u
    Rv = vt + f * u + G * ey + r * v
    Reta = et + (Hx * u + H * ux) + (Hy * v + H * vy)
    return Ru, Rv, Reta


# -------------------------------------------------------------------------
# Coastal PINN: (x,y,t) -> (u,v,eta), with optional Fourier features (tidal
# periodicity in time) and an optional trainable drag r>=0.
# -------------------------------------------------------------------------
class CoastalPINN(nn.Module):
    def __init__(self, x_scale, y_scale, t_scale, u_scale, eta_scale,
                 width=64, depth=4, n_freq=None, train_drag=False, r_init=1e-6):
        super().__init__()
        dt = torch.get_default_dtype()
        self.register_buffer("input_scale",
                             torch.tensor([x_scale, y_scale, t_scale], dtype=dt))
        self.register_buffer("output_scale",
                             torch.tensor([u_scale, u_scale, eta_scale], dtype=dt))
        self.n_freq = n_freq
        if n_freq is not None:
            fx, fy, ft = n_freq
            self.register_buffer("fx", math.pi * torch.arange(1, fx + 1, dtype=dt))
            self.register_buffer("fy", math.pi * torch.arange(1, fy + 1, dtype=dt))
            self.register_buffer("ft", math.pi * torch.arange(1, ft + 1, dtype=dt))
            nin = 3 + 2 * (fx + fy + ft)
        else:
            nin = 3
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(nin, width), nn.Tanh()]
            nin = width
        layers.append(nn.Linear(nin, 3))
        self.net = nn.Sequential(*layers)
        self.train_drag = train_drag
        if train_drag:
            self.log_r = nn.Parameter(torch.tensor(math.log(r_init), dtype=dt))

    @property
    def r(self):
        return torch.exp(self.log_r) if self.train_drag else 0.0

    def _embed(self, z):
        if self.n_freq is None:
            return z
        feats = [z]
        for col, fr in ((z[:, 0:1], self.fx), (z[:, 1:2], self.fy), (z[:, 2:3], self.ft)):
            a = col * fr
            feats += [torch.sin(a), torch.cos(a)]
        return torch.cat(feats, dim=1)

    def forward(self, X):
        z = X / self.input_scale
        z = z.clone()
        z[:, 2] = 2.0 * z[:, 2] - 1.0     # t in [0,t_scale] -> [-1,1]
        q = self.net(self._embed(z))
        out = q * self.output_scale
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
