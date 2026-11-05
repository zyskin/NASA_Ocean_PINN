#!/usr/bin/env python3
"""
physics.py

Shared physics for the linear equatorial beta-plane shallow-water system:

    u_t - beta*y*v + Phi_x = 0
    v_t + beta*y*u + Phi_y = 0
    Phi_t + c^2 (u_x + v_y) = 0

with beta = 2*Omega/R_E.

Single source of truth for:
  * the PINN model (EquatorialPINN) and residual operator (pinn_residuals),
  * analytic equatorial wave solutions (Kelvin + general Matsuno/Rossby modes),
  * autograd helpers.

Signs and dimensional factors are identical to the verified
equatorial_kelvin_pinn_test.py. All PDE work is in physical SI:
x,y in m, t in s, u,v in m/s, Phi = g*eta in m^2/s^2.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from config import OMEGA, R_E, G, BETA   # noqa: F401,E402


# -------------------------------------------------------------------------
# autograd helper
# -------------------------------------------------------------------------
def grad(q, X):
    # A field with no grad_fn (e.g. v identically zero) has zero derivatives.
    if not q.requires_grad:
        return torch.zeros_like(X)
    g = torch.autograd.grad(
        q, X, grad_outputs=torch.ones_like(q),
        create_graph=True, retain_graph=True, allow_unused=True,
    )[0]
    return torch.zeros_like(X) if g is None else g


# -------------------------------------------------------------------------
# Hermite functions psi_n(z) = H_n(z) exp(-z^2/2)
# -------------------------------------------------------------------------
def _hermite(n, z):
    if n < 0:
        return torch.zeros_like(z)
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return 2.0 * z
    Hnm1 = torch.ones_like(z)
    Hn = 2.0 * z
    for m in range(1, n):
        Hnp1 = 2.0 * z * Hn - 2.0 * m * Hnm1
        Hnm1, Hn = Hn, Hnp1
    return Hn


def _psi(n, z):
    if n < 0:
        return torch.zeros_like(z)
    return _hermite(n, z) * torch.exp(-0.5 * z * z)


# -------------------------------------------------------------------------
# Analytic equatorial Kelvin wave (eastward, v == 0)
#   Phi = A exp(-beta y^2/(2c)) cos[k(x - c t)],  u = Phi/c,  v = 0
# -------------------------------------------------------------------------
def kelvin_solution(X, c, A, k):
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    envelope = torch.exp(-BETA * y * y / (2.0 * c))
    phase = k * (x - c * t)
    phi = A * envelope * torch.cos(phase)
    u = phi / c
    v = torch.zeros_like(phi)
    return u, v, phi


# -------------------------------------------------------------------------
# Equatorial dispersion relation (SI). Cubic in omega:
#   omega^3 - c^2 (k^2 + (2n+1) beta/c) omega - c^2 beta k = 0
# Roots: eastward gravity, westward gravity, Rossby (low-|omega|).
# -------------------------------------------------------------------------
def dispersion_roots(k, n, c):
    coeffs = [1.0, 0.0,
              -c * c * (k * k + (2 * n + 1) * BETA / c),
              -c * c * BETA * k]
    roots = [r.real for r in np.roots(coeffs) if abs(r.imag) < 1e-9 * (abs(r.real) + 1)]
    return sorted(roots)


def rossby_dispersion(k, n, c):
    """Low-frequency (Rossby) root; westward (negative) for k>0."""
    roots = dispersion_roots(k, n, c)
    roots.sort(key=lambda w: abs(w))
    return float(roots[0])


# -------------------------------------------------------------------------
# Exact Matsuno (1966) mode, meridional index n>=1, in SI units.
#
# Nondimensionalise with L=sqrt(c/beta), velocity c, geopotential c^2,
# time L/c. With v_hat = psi_n(z) and the Hermite ladder relations
#     y psi_n = (1/2) psi_{n+1} + n psi_{n-1}
#     psi_n'  = -(1/2) psi_{n+1} + n psi_{n-1}
# the spectral equations give (k,w NONDIMENSIONAL; z=y/L; theta=k x - w t):
#     v' = psi_n cos(theta)
#     u' = [ (w+k)/2 psi_{n+1} + n(w-k) psi_{n-1} ] / (k^2 - w^2) sin(theta)
#     phi'=[ (w+k)/2 psi_{n+1} - n(w-k) psi_{n-1} ] / (k^2 - w^2) sin(theta)
# Dimensional: u=c u', v=c v', phi=c^2 phi'.
# Verified by autograd substitution in tests/rossby_analytic_test.py.
# -------------------------------------------------------------------------
def matsuno_mode(X, c, A, k, n=1, omega=None, branch="rossby"):
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    if omega is None:
        if branch == "rossby":
            omega = rossby_dispersion(k, n, c)
        else:
            roots = dispersion_roots(k, n, c)
            omega = max(roots) if branch == "egrav" else min(roots)

    L = math.sqrt(c / BETA)
    z = y / L
    kp = k * L
    wp = omega / math.sqrt(BETA * c)

    theta = k * x - omega * t
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    psi_n = _psi(n, z)
    psi_nm1 = _psi(n - 1, z)
    psi_np1 = _psi(n + 1, z)

    denom = (kp * kp - wp * wp)
    pre = 0.5 * (wp + kp)
    cm1 = n * (wp - kp)

    up = (pre * psi_np1 + cm1 * psi_nm1) / denom * sin_t
    php = (pre * psi_np1 - cm1 * psi_nm1) / denom * sin_t
    vp = psi_n * cos_t

    u = c * up
    v = c * vp
    phi = (c * c) * php

    eps = 1e-30
    s = A / (phi.abs().max().item() + eps)
    return u * s, v * s, phi * s, float(omega)


def rossby_solution(X, c, A, k, n=1, omega=None):
    return matsuno_mode(X, c, A, k, n=n, omega=omega, branch="rossby")


# -------------------------------------------------------------------------
# PINN model: (x,y,t) -> (u,v,Phi), trainable c>0.
# -------------------------------------------------------------------------
class EquatorialPINN(nn.Module):
    """
    (x,y,t) -> (u,v,Phi), trainable c>0.

    Optional per-axis Fourier feature embedding (Tancik et al. 2020) defeats the
    spectral bias of a plain tanh MLP, which otherwise cannot represent fields
    with many oscillations across the record (e.g. weekly SSH over >1 yr). Set
    n_freq=(fx,fy,ft) harmonics per normalised axis; n_freq=None recovers the
    plain architecture used in equatorial_kelvin_pinn_test.py. Features are
    smooth functions of X, so PDE autograd remains exact.
    """

    def __init__(self, x_scale, y_scale, t_scale,
                 u_scale, phi_scale, c_init=1.0, width=64, depth=4,
                 n_freq=None):
        super().__init__()
        dt = torch.get_default_dtype()
        self.register_buffer("input_scale",
                             torch.tensor([x_scale, y_scale, t_scale], dtype=dt))
        self.register_buffer("output_scale",
                             torch.tensor([u_scale, u_scale, phi_scale], dtype=dt))

        self.n_freq = n_freq
        if n_freq is not None:
            fx, fy, ft = n_freq
            # linear harmonics m=1..f per axis; pi*m*z gives m half-cycles over [-1,1]
            self.register_buffer("freq_x", math.pi * torch.arange(1, fx + 1, dtype=dt))
            self.register_buffer("freq_y", math.pi * torch.arange(1, fy + 1, dtype=dt))
            self.register_buffer("freq_t", math.pi * torch.arange(1, ft + 1, dtype=dt))
            nin = 3 + 2 * (fx + fy + ft)
        else:
            nin = 3

        layers = []
        for _ in range(depth):
            layers += [nn.Linear(nin, width), nn.Tanh()]
            nin = width
        layers.append(nn.Linear(nin, 3))
        self.net = nn.Sequential(*layers)
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init), dtype=dt))

    @property
    def c(self):
        return torch.exp(self.log_c)

    def _embed(self, z):
        if self.n_freq is None:
            return z
        zx, zy, zt = z[:, 0:1], z[:, 1:2], z[:, 2:3]
        feats = [z]
        for col, freq in ((zx, self.freq_x), (zy, self.freq_y), (zt, self.freq_t)):
            arg = col * freq            # (N, f)
            feats.append(torch.sin(arg))
            feats.append(torch.cos(arg))
        return torch.cat(feats, dim=1)

    def forward(self, X):
        z = X / self.input_scale
        z = z.clone()
        z[:, 2] = 2.0 * z[:, 2] - 1.0     # t in [0,t_scale] -> [-1,1]
        q = self.net(self._embed(z))
        out = q * self.output_scale
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def pinn_residuals(model, X, c=None):
    """PDE residuals in physical SI coordinates (model.c used unless c given)."""
    X = X.clone().detach().requires_grad_(True)
    u, v, phi = model(X)
    gu, gv, gp = grad(u, X), grad(v, X), grad(phi, X)
    ux, ut = gu[:, 0:1], gu[:, 2:3]
    vy, vt = gv[:, 1:2], gv[:, 2:3]
    px, py, pt = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    y = X[:, 1:2]
    f = BETA * y
    cc = (model.c if c is None else c)
    Ru = ut - f * v + px
    Rv = vt + f * u + py
    Rphi = pt + cc * cc * (ux + vy)
    return Ru, Rv, Rphi
