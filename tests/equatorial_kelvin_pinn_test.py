#!/usr/bin/env python3
"""
equatorial_kelvin_pinn_test.py

Two tests for a physics-informed neural network (PINN) for the linear
equatorial shallow-water equations:

    u_t - beta*y*v + Phi_x = 0
    v_t + beta*y*u + Phi_y = 0
    Phi_t + c^2 (u_x + v_y) = 0

1. Verify the analytic equatorial Kelvin wave by automatic differentiation.
2. Train a small PINN using ONLY synthetic Phi = g*eta observations and PDE
   residuals, while treating c as an unknown positive trainable parameter.

The synthetic truth is
    Phi = A exp[-beta*y^2/(2c)] cos[k(x-c t)]
    u   = Phi/c
    v   = 0.

The code saves:
    kelvin_pinn_checkpoint.pt
    kelvin_pinn_history.csv
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# -------------------------
# Physical constants
# -------------------------
OMEGA = 7.292115e-5          # rad/s
R_E = 6.371e6                # m
G = 9.81                     # m/s^2
BETA = 2.0 * OMEGA / R_E     # 1/(m s)

# -------------------------
# Synthetic truth
# -------------------------
C_TRUE = 2.5                  # m/s
A = 0.5                       # geopotential amplitude, m^2/s^2
WAVELENGTH = 8.0e6            # m
K = 2.0 * math.pi / WAVELENGTH

# Domain
XMAX = 4.0e6                  # m
YMAX = 9.0e5                  # m
TMAX = 30.0 * 86400.0         # s

# Characteristic output scales used only to condition the neural network.
U_SCALE = 0.25                # m/s
PHI_SCALE = A                 # m^2/s^2

# Characteristic residual scales used only to balance the three residual losses.
RU_SCALE = 4.0e-7             # m/s^2
RV_SCALE = 1.5e-6             # m/s^2
RPHI_SCALE = 1.0e-6           # m^2/s^3


def grad(q, X):
    return torch.autograd.grad(
        q, X,
        grad_outputs=torch.ones_like(q),
        create_graph=True,
        retain_graph=True,
    )[0]


def sample_X(n, device):
    X = torch.empty(n, 3, device=device)
    X[:, 0].uniform_(-XMAX, XMAX)
    X[:, 1].uniform_(-YMAX, YMAX)
    X[:, 2].uniform_(0.0, TMAX)
    return X


def kelvin_solution(X, c=C_TRUE):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]

    envelope = torch.exp(-BETA * y * y / (2.0 * c))
    phase = K * (x - c * t)

    phi = A * envelope * torch.cos(phase)
    u = phi / c
    v = torch.zeros_like(phi)
    return u, v, phi


def analytic_residuals(X, c=C_TRUE):
    """
    Residuals of the exact Kelvin solution, computed with autograd.
    """
    X = X.clone().detach().requires_grad_(True)
    u, v, phi = kelvin_solution(X, c)

    gu = grad(u, X)
    gp = grad(phi, X)

    ux = gu[:, 0:1]
    ut = gu[:, 2:3]

    px = gp[:, 0:1]
    py = gp[:, 1:2]
    pt = gp[:, 2:3]

    y = X[:, 1:2]

    # Since v is identically zero, v_t = v_y = 0 exactly.
    Ru = ut + px
    Rv = BETA * y * u + py
    Rphi = pt + c * c * ux

    return Ru, Rv, Rphi


class EquatorialPINN(nn.Module):
    def __init__(self, c_init=1.0, width=48, depth=3):
        super().__init__()

        self.register_buffer(
            "input_scale",
            torch.tensor([XMAX, YMAX, TMAX], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "output_scale",
            torch.tensor([U_SCALE, U_SCALE, PHI_SCALE],
                         dtype=torch.get_default_dtype())
        )

        layers = []
        nin = 3
        for _ in range(depth):
            layers.append(nn.Linear(nin, width))
            layers.append(nn.Tanh())
            nin = width
        layers.append(nn.Linear(nin, 3))
        self.net = nn.Sequential(*layers)

        # c > 0 automatically.
        self.log_c = nn.Parameter(
            torch.tensor(math.log(c_init), dtype=torch.get_default_dtype())
        )

    @property
    def c(self):
        return torch.exp(self.log_c)

    def forward(self, X):
        z = X / self.input_scale
        z = z.clone()

        # x,y are already centered about zero. Put t in [-1,1].
        z[:, 2] = 2.0 * z[:, 2] - 1.0

        q = self.net(z)
        out = q * self.output_scale

        u = out[:, 0:1]
        v = out[:, 1:2]
        phi = out[:, 2:3]
        return u, v, phi


def pinn_residuals(model, X):
    """
    PDE residuals in physical SI coordinates.
    Input normalization happens inside model(), but autograd differentiates
    with respect to the physical X, so metric factors are retained.
    """
    X = X.clone().detach().requires_grad_(True)
    u, v, phi = model(X)

    gu = grad(u, X)
    gv = grad(v, X)
    gp = grad(phi, X)

    ux = gu[:, 0:1]
    ut = gu[:, 2:3]

    vy = gv[:, 1:2]
    vt = gv[:, 2:3]

    px = gp[:, 0:1]
    py = gp[:, 1:2]
    pt = gp[:, 2:3]

    y = X[:, 1:2]
    f = BETA * y

    Ru = ut - f * v + px
    Rv = vt + f * u + py
    Rphi = pt + model.c**2 * (ux + vy)

    return Ru, Rv, Rphi


def run_analytic_test(device):
    torch.manual_seed(0)
    X = sample_X(5000, device)

    Ru, Rv, Rphi = analytic_residuals(X)

    print("\nAnalytic Kelvin-wave residual test")
    for name, r in [("Ru", Ru), ("Rv", Rv), ("Rphi", Rphi)]:
        maxabs = r.abs().max().item()
        rms = torch.sqrt(torch.mean(r * r)).item()
        print(f"{name:5s}: max|R| = {maxabs:.3e}, RMS = {rms:.3e}")


def train_phi_only_pinn(device):
    """
    Train from Phi observations only. No u or v labels are supplied.
    """
    torch.manual_seed(7)

    n_data = 900
    n_col = 1800

    Xd = sample_X(n_data, device)
    phi_data = kelvin_solution(Xd)[2].detach()

    Xc = sample_X(n_col, device)

    model = EquatorialPINN(c_init=1.0, width=48, depth=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)

    history = []

    print("\nAdam training: Phi observations only")
    t0 = time.time()

    for it in range(2501):
        optimizer.zero_grad()

        _, _, phi_pred = model(Xd)
        loss_data = torch.mean(((phi_pred - phi_data) / PHI_SCALE)**2)

        Ru, Rv, Rphi = pinn_residuals(model, Xc)
        loss_pde = (
            torch.mean((Ru / RU_SCALE)**2)
            + torch.mean((Rv / RV_SCALE)**2)
            + torch.mean((Rphi / RPHI_SCALE)**2)
        )

        # Very weak regularization only to discourage arbitrary large
        # unconstrained velocities during early training.
        u_c, v_c, _ = model(Xc)
        loss_velreg = 1.0e-4 * (
            torch.mean((u_c / U_SCALE)**2)
            + torch.mean((v_c / U_SCALE)**2)
        )

        loss = loss_data + loss_pde + loss_velreg
        loss.backward()
        optimizer.step()

        if it % 250 == 0:
            row = [
                it,
                loss.item(),
                loss_data.item(),
                loss_pde.item(),
                model.c.detach().item(),
            ]
            history.append(row)
            print(
                f"{it:5d}  loss={row[1]:.3e}  "
                f"data={row[2]:.3e}  pde={row[3]:.3e}  "
                f"c={row[4]:.6f} m/s"
            )

    print(f"Adam elapsed: {time.time()-t0:.2f} s")

    # Short full-batch L-BFGS refinement.
    print("\nL-BFGS refinement")
    optimizer2 = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=50,
        max_eval=60,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-11,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    calls = [0]

    def closure():
        optimizer2.zero_grad()

        _, _, phi_pred = model(Xd)
        loss_data = torch.mean(((phi_pred - phi_data) / PHI_SCALE)**2)

        Ru, Rv, Rphi = pinn_residuals(model, Xc)
        loss_pde = (
            torch.mean((Ru / RU_SCALE)**2)
            + torch.mean((Rv / RV_SCALE)**2)
            + torch.mean((Rphi / RPHI_SCALE)**2)
        )

        u_c, v_c, _ = model(Xc)
        loss_velreg = 1.0e-4 * (
            torch.mean((u_c / U_SCALE)**2)
            + torch.mean((v_c / U_SCALE)**2)
        )

        loss = loss_data + loss_pde + loss_velreg
        loss.backward()
        calls[0] += 1
        return loss

    t1 = time.time()
    optimizer2.step(closure)
    print(f"L-BFGS function calls: {calls[0]}")
    print(f"L-BFGS elapsed: {time.time()-t1:.2f} s")

    c_est = model.c.detach().item()

    # Independent held-out test points.
    torch.manual_seed(123)
    Xt = sample_X(5000, device)

    with torch.no_grad():
        u_pred, v_pred, phi_pred = model(Xt)
        u_true, v_true, phi_true = kelvin_solution(Xt)

        rmse_phi = torch.sqrt(torch.mean((phi_pred - phi_true)**2)).item()
        rmse_u = torch.sqrt(torch.mean((u_pred - u_true)**2)).item()
        rmse_v = torch.sqrt(torch.mean((v_pred - v_true)**2)).item()

        rms_phi_true = torch.sqrt(torch.mean(phi_true**2)).item()
        rms_u_true = torch.sqrt(torch.mean(u_true**2)).item()

    Ru, Rv, Rphi = pinn_residuals(model, Xt)

    print("\nHeld-out diagnostics")
    print(f"c true                  = {C_TRUE:.6f} m/s")
    print(f"c inferred              = {c_est:.6f} m/s")
    print(f"relative c error         = {100*abs(c_est-C_TRUE)/C_TRUE:.4f} %")
    print(f"equivalent depth true    = {C_TRUE**2/G:.6f} m")
    print(f"equivalent depth inferred= {c_est**2/G:.6f} m")
    print(f"Phi relative RMSE        = {100*rmse_phi/rms_phi_true:.4f} %")
    print(f"u relative RMSE          = {100*rmse_u/rms_u_true:.4f} %")
    print(f"v RMSE                   = {rmse_v:.6e} m/s")
    print(f"Ru RMS                   = {torch.sqrt(torch.mean(Ru**2)).item():.6e}")
    print(f"Rv RMS                   = {torch.sqrt(torch.mean(Rv**2)).item():.6e}")
    print(f"Rphi RMS                 = {torch.sqrt(torch.mean(Rphi**2)).item():.6e}")

    # Save reproducibility artifacts.
    with open("kelvin_pinn_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adam_step", "total_loss", "data_loss", "pde_loss", "c_m_per_s"])
        w.writerows(history)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "c_true": C_TRUE,
            "c_est": c_est,
            "constants": {
                "Omega": OMEGA,
                "R_E": R_E,
                "beta": BETA,
                "g": G,
                "A": A,
                "wavelength": WAVELENGTH,
            },
        },
        "kelvin_pinn_checkpoint.pt",
    )

    return model


def main():
    # Float32 makes the PINN training fast. The analytic identity is still
    # checked to near single-precision limits. Change to float64 if desired.
    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"beta   = {BETA:.12e} 1/(m s)")

    run_analytic_test(device)
    train_phi_only_pinn(device)


if __name__ == "__main__":
    main()
