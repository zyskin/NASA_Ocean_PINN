import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

FILE = "data_processed/ssh_oscar_weekly_pacific.nc"

C_K = 2.45               # m/s, independent Kelvin reference
BETA = 2.289e-11         # m^-1 s^-1
G = 9.81
R_EARTH = 6371e3

# Analyze the cleaner central/eastern Pacific section.
LON0 = -170.0
LON1 = -100.0

# Expected n=1 Rossby structure tested mainly here.
PARITY_LAT0 = 2.0
PARITY_LAT1 = 8.0

# Periods kept in Rossby filter.
PERIOD_MIN_D = 20.0
PERIOD_MAX_D = 180.0

# Zonal wavelengths kept.
WAVELENGTH_MIN_KM = 1800.0
WAVELENGTH_MAX_KM = 10000.0

os.makedirs("figures", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)


# ============================================================
# LOAD
# ============================================================

ds = xr.open_dataset(FILE)

ds = ds.sel(
    lon=slice(LON0, LON1)
)

print()
print(ds)

dates0 = pd.DatetimeIndex(
    ds.time.values
)

print()
print("input dates:", len(dates0))
print("range:", dates0[0], "to", dates0[-1])


# ============================================================
# PUT DATA ON AN EXACTLY REGULAR WEEKLY TIME GRID
# ============================================================

dt0 = np.diff(
    dates0.values
) / np.timedelta64(1, "D")

median_dt = float(
    np.median(dt0)
)

print(
    "median original time spacing:",
    median_dt,
    "days"
)

# Use approximately weekly spacing.
DT_DAYS = 7.0

newtime = np.arange(
    ds.time.values[0],
    ds.time.values[-1] + np.timedelta64(1, "D"),
    np.timedelta64(7, "D")
)

ds = ds.interp(
    time=newtime
)

dates = pd.DatetimeIndex(
    ds.time.values
)

print(
    "regularized maps:",
    len(dates)
)


# ============================================================
# NUMERIC ARRAYS
# ============================================================

lat = ds.lat.values
lon = ds.lon.values

ETA = ds.ssha.values
U   = ds.u.values
V   = ds.v.values
UG  = ds.ug.values
VG  = ds.vg.values

nt, ny, nx = ETA.shape

print(
    "cube shape time,lat,lon:",
    ETA.shape
)


# ============================================================
# FILL SMALL MISSING VALUES IN TIME/LONGITUDE
#
# Tropical ocean should be mostly complete. We only use this
# to make FFT possible.
# ============================================================

def fill_cube(A):

    B = np.asarray(
        A,
        dtype=float
    ).copy()

    # Along longitude first.
    xx = np.arange(nx)

    for it in range(nt):
        for iy in range(ny):

            y = B[it, iy, :]
            good = np.isfinite(y)

            if good.sum() >= 2:
                B[it, iy, :] = np.interp(
                    xx,
                    xx[good],
                    y[good]
                )

    # Along time second.
    tt = np.arange(nt)

    for iy in range(ny):
        for ix in range(nx):

            y = B[:, iy, ix]
            good = np.isfinite(y)

            if good.sum() >= 2:
                B[:, iy, ix] = np.interp(
                    tt,
                    tt[good],
                    y[good]
                )

    return B


ETA = fill_cube(ETA)
U   = fill_cube(U)
V   = fill_cube(V)
UG  = fill_cube(UG)
VG  = fill_cube(VG)


# ============================================================
# REMOVE TIME MEAN + LINEAR TREND AT EVERY GRID POINT
# ============================================================

def detrend_cube(A):

    B = A.copy()

    t = np.arange(nt, dtype=float)

    for iy in range(ny):
        for ix in range(nx):

            y = B[:, iy, ix]

            good = np.isfinite(y)

            if good.sum() < 4:
                B[:, iy, ix] = 0.0
                continue

            p = np.polyfit(
                t[good],
                y[good],
                1
            )

            B[:, iy, ix] = (
                y - np.polyval(p, t)
            )

    # Remove spatially uniform zonal anomaly at each time/lat.
    B -= np.nanmean(
        B,
        axis=2,
        keepdims=True
    )

    return B


eta = detrend_cube(ETA)
u   = detrend_cube(U)
v   = detrend_cube(V)
ug  = detrend_cube(UG)
vg  = detrend_cube(VG)


# ============================================================
# FFT GEOMETRY
# ============================================================

dt_s = DT_DAYS * 86400.0

dlon = float(
    np.median(np.diff(lon))
)

# representative equatorial spacing
dx_m = (
    np.deg2rad(dlon)
    * R_EARTH
)

ft = np.fft.fftfreq(
    nt,
    d=dt_s
)                       # cycles / second

kx = np.fft.fftfreq(
    nx,
    d=dx_m
)                       # cycles / meter


# physical temporal frequency:
#
# numpy FFT basis is exp[-i 2pi(ft t + k x)]
# while wave convention is exp[i(kx - omega t)].
#
# Hence:
#
#     f_phys = -ft
#
f_phys = -ft


# ============================================================
# THEORETICAL n=1 ROSSBY DISPERSION RELATION
# ============================================================

L = np.sqrt(
    C_K / BETA
)

print()
print(
    "equatorial deformation scale L:",
    L/1000,
    "km"
)

K_RAD = (
    2*np.pi*kx
)

omega_R = (
    -BETA*K_RAD
    /
    (
        K_RAD**2
        + 3.0/L**2
    )
)

f_R = (
    omega_R
    /
    (2*np.pi)
)                       # cycles / sec

f_R_day = (
    f_R * 86400.0
)


# ============================================================
# BUILD ROSSBY SPECTRAL MASK
# ============================================================

FGRID = (
    f_phys[:, None]
)

KGRID = (
    kx[None, :]
)

FRGRID = (
    f_R[None, :]
)


# Frequency resolution.
df_day = (
    1.0
    /
    (nt * DT_DAYS)
)

# Allow roughly 1.5 FFT frequency bins around theory.
TOL_DAY = max(
    1.5 * df_day,
    0.003
)

print(
    "frequency resolution:",
    df_day,
    "cycles/day"
)

print(
    "Rossby filter half-width:",
    TOL_DAY,
    "cycles/day"
)


fgrid_day = (
    FGRID * 86400.0
)

frgrid_day = (
    FRGRID * 86400.0
)


# Period range.
frequency_ok = (
    (np.abs(fgrid_day) >= 1.0/PERIOD_MAX_D)
    &
    (np.abs(fgrid_day) <= 1.0/PERIOD_MIN_D)
)


# Wavelength range.
kabs = np.abs(KGRID)

kmin = 1.0 / (
    WAVELENGTH_MAX_KM * 1000.0
)

kmax = 1.0 / (
    WAVELENGTH_MIN_KM * 1000.0
)

wavelength_ok = (
    (kabs >= kmin)
    &
    (kabs <= kmax)
)


dispersion_ok = (
    np.abs(
        fgrid_day
        -
        frgrid_day
    )
    <= TOL_DAY
)


MASK = (
    frequency_ok
    &
    wavelength_ok
    &
    dispersion_ok
)

print(
    "Rossby spectral bins retained:",
    int(MASK.sum())
)


# ============================================================
# WINDOW + FFT
# ============================================================

wt = np.hanning(nt)
wx = np.hanning(nx)

window = (
    wt[:, None, None]
    *
    wx[None, None, :]
)


def fft_field(A):

    return np.fft.fft2(
        A * window,
        axes=(0, 2)
    )


FETA = fft_field(eta)
FU   = fft_field(u)
FV   = fft_field(v)
FUG  = fft_field(ug)
FVG  = fft_field(vg)


# ============================================================
# FILTER ALONG ROSSBY DISPERSION BAND
# ============================================================

MASK3 = MASK[:, None, :]


def inverse_filter(F):

    return np.fft.ifft2(
        F * MASK3,
        axes=(0, 2)
    ).real


etaR = inverse_filter(FETA)
uR   = inverse_filter(FU)
vR   = inverse_filter(FV)

ugR  = inverse_filter(FUG)
vgR  = inverse_filter(FVG)


# ============================================================
# FIND DOMINANT ROSSBY SPECTRAL BIN FROM SSH
# ============================================================

north = (
    (lat >= 3.0)
    &
    (lat <= 8.0)
)

south = (
    (lat >= -8.0)
    &
    (lat <= -3.0)
)


# Construct symmetric SSH Fourier amplitude:
FETA_SYM = (
    np.mean(
        FETA[:, north, :],
        axis=1
    )
    +
    np.mean(
        FETA[:, south, :],
        axis=1
    )
) / 2.0


POWER_SYM = (
    np.abs(FETA_SYM)**2
)


# Search only positive zonal k branch;
# negative-k conjugate carries same information.
candidate = (
    MASK
    &
    (KGRID > 0)
)

Psearch = np.where(
    candidate,
    POWER_SYM,
    -np.inf
)

it_peak, ik_peak = np.unravel_index(
    np.nanargmax(Psearch),
    Psearch.shape
)


f_peak = f_phys[it_peak]
k_peak = kx[ik_peak]

c_peak = (
    f_peak / k_peak
)

period_peak_days = (
    1.0 / abs(f_peak) / 86400.0
)

wavelength_peak_km = (
    1.0 / abs(k_peak) / 1000.0
)


print()
print("DOMINANT SSH ROSSBY-BAND BIN")
print("----------------------------")
print(
    "phase speed:",
    c_peak,
    "m/s"
)
print(
    "period:",
    period_peak_days,
    "days"
)
print(
    "wavelength:",
    wavelength_peak_km,
    "km"
)
print(
    "theory frequency:",
    f_R_day[ik_peak],
    "cycles/day"
)
print(
    "observed bin frequency:",
    f_peak*86400.0,
    "cycles/day"
)


# ============================================================
# PARITY TEST
#
# n=1 Rossby expectation:
#
#   eta(-y) = +eta(y)   even
#   u(-y)   = +u(y)     even
#   v(-y)   = -v(y)     odd
#
# ============================================================

def parity_energy(A):

    Eeven = 0.0
    Eodd = 0.0

    used = []

    for jp, yp in enumerate(lat):

        if not (
            PARITY_LAT0
            <= yp
            <= PARITY_LAT1
        ):
            continue

        jm = int(
            np.argmin(
                np.abs(lat + yp)
            )
        )

        if abs(
            lat[jm] + yp
        ) > 0.26:
            continue

        plus = A[:, jp, :]
        minus = A[:, jm, :]

        even = 0.5 * (
            plus + minus
        )

        odd = 0.5 * (
            plus - minus
        )

        Eeven += np.nansum(
            even**2
        )

        Eodd += np.nansum(
            odd**2
        )

        used.append(
            (yp, lat[jm])
        )

    total = Eeven + Eodd

    return (
        Eeven / total,
        Eodd / total,
        used
    )


eta_even, eta_odd, pairs = parity_energy(
    etaR
)

u_even, u_odd, _ = parity_energy(
    uR
)

v_even, v_odd, _ = parity_energy(
    vR
)


print()
print("ROSSBY-BAND PARITY")
print("------------------")

print(
    "SSH even fraction:",
    eta_even,
    " expected HIGH"
)

print(
    "u even fraction:",
    u_even,
    " expected HIGH"
)

print(
    "v odd fraction:",
    v_odd,
    " expected HIGH"
)


# ============================================================
# DIRECT MIRROR CORRELATIONS
# ============================================================

def mirror_corr(A, odd=False):

    aa = []
    bb = []

    for jp, yp in enumerate(lat):

        if not (
            PARITY_LAT0
            <= yp
            <= PARITY_LAT1
        ):
            continue

        jm = int(
            np.argmin(
                np.abs(lat + yp)
            )
        )

        if abs(
            lat[jm] + yp
        ) > 0.26:
            continue

        x = A[:, jp, :].ravel()
        y = A[:, jm, :].ravel()

        if odd:
            y = -y

        good = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        aa.append(
            x[good]
        )
        bb.append(
            y[good]
        )

    a = np.concatenate(aa)
    b = np.concatenate(bb)

    return np.corrcoef(
        a,
        b
    )[0, 1]


corr_eta = mirror_corr(
    etaR,
    odd=False
)

corr_u = mirror_corr(
    uR,
    odd=False
)

corr_v = mirror_corr(
    vR,
    odd=True
)


print()
print("MIRROR CORRELATIONS")
print("-------------------")
print(
    "corr[eta(+y), eta(-y)] =",
    corr_eta
)
print(
    "corr[u(+y), u(-y)] =",
    corr_u
)
print(
    "corr[v(+y), -v(-y)] =",
    corr_v
)


# ============================================================
# HOW GEOSTROPHIC IS THE FILTERED SIGNAL?
# ============================================================

mband = (
    np.abs(lat) <= 8.0
)

Etot = np.mean(
    uR[:, mband, :]**2
    +
    vR[:, mband, :]**2
)

Egeo = np.mean(
    ugR[:, mband, :]**2
    +
    vgR[:, mband, :]**2
)

print()
print("FILTERED CURRENT ENERGY")
print("-----------------------")
print(
    "total-current RMS:",
    np.sqrt(Etot),
    "m/s"
)
print(
    "geostrophic RMS:",
    np.sqrt(Egeo),
    "m/s"
)
print(
    "geostrophic/total RMS ratio:",
    np.sqrt(Egeo/Etot)
)


# ============================================================
# PLOT 1:
# k-omega POWER OF SSH AND OSCAR CURRENT
# ============================================================

# Current spectral power:
PCUR = np.mean(
    np.abs(FU)**2
    +
    np.abs(FV)**2,
    axis=1
)

# Shift for plotting.
fplot = np.fft.fftshift(
    f_phys * 86400.0
)

kplot = np.fft.fftshift(
    kx * 1.0e7
)

Peta_plot = np.fft.fftshift(
    POWER_SYM,
    axes=(0, 1)
)

Pcur_plot = np.fft.fftshift(
    PCUR,
    axes=(0, 1)
)

# positive k display
kmask = (
    (kplot > 0)
    &
    (kplot <= 7)
)

fmask = (
    (fplot >= -0.06)
    &
    (fplot <= 0.06)
)


fig, ax = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


def logpow(P):
    Q = P[np.ix_(
        fmask,
        kmask
    )]

    floor = (
        np.nanmax(Q) * 1e-7
    )

    return np.log10(
        Q + floor
    )


# theoretical curve on positive k
kx_pos = (
    kplot[kmask] / 1.0e7
)

krad_pos = (
    2*np.pi*kx_pos
)

omega_theory = (
    -BETA * krad_pos
    /
    (
        krad_pos**2
        +
        3.0/L**2
    )
)

f_theory = (
    omega_theory
    /
    (2*np.pi)
    * 86400.0
)


p0 = ax[0].pcolormesh(
    kplot[kmask],
    fplot[fmask],
    logpow(Peta_plot),
    shading="auto",
    cmap="magma"
)

ax[0].plot(
    kplot[kmask],
    f_theory,
    "w--",
    lw=2
)

ax[0].set_title(
    "NASA SSH symmetric component"
)

ax[0].set_xlabel(
    "Zonal wavenumber [cycles / 10,000 km]"
)

ax[0].set_ylabel(
    "Frequency [cycles/day]"
)

fig.colorbar(
    p0,
    ax=ax[0],
    label="log10 power"
)


p1 = ax[1].pcolormesh(
    kplot[kmask],
    fplot[fmask],
    logpow(Pcur_plot),
    shading="auto",
    cmap="magma"
)

ax[1].plot(
    kplot[kmask],
    f_theory,
    "w--",
    lw=2
)

ax[1].set_title(
    "OSCAR total-current power"
)

ax[1].set_xlabel(
    "Zonal wavenumber [cycles / 10,000 km]"
)

ax[1].set_ylabel(
    "Frequency [cycles/day]"
)

fig.colorbar(
    p1,
    ax=ax[1],
    label="log10 power"
)

fig.suptitle(
    "Rossby dispersion-band comparison: SSH and water velocity"
)

plt.tight_layout()

plt.savefig(
    "figures/oscar_rossby_komega_comparison.png",
    dpi=240
)

plt.close()


# ============================================================
# PLOT 2:
# PARITY TEST
# ============================================================

labels = [
    "SSH even",
    "u even",
    "v odd"
]

values = [
    eta_even,
    u_even,
    v_odd
]


fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    labels,
    values
)

ax.axhline(
    0.5,
    color="black",
    ls="--",
    lw=1
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Fraction of Rossby-band energy\n"
    "with expected parity"
)

ax.set_title(
    "n=1 equatorial Rossby parity test"
)

for bar, val in zip(
    bars,
    values
):
    ax.text(
        bar.get_x()
        + bar.get_width()/2,
        val + 0.025,
        f"{val:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "figures/oscar_rossby_parity.png",
    dpi=240
)

plt.close()


# ============================================================
# PLOT 3:
# ROSSBY-FILTERED VECTOR SNAPSHOT
# ============================================================

speedR = np.sqrt(
    uR**2
    +
    vR**2
)

band = (
    (np.abs(lat) >= 2)
    &
    (np.abs(lat) <= 8)
)

rms_time = np.sqrt(
    np.mean(
        speedR[:, band, :]**2,
        axis=(1, 2)
    )
)

it_show = int(
    np.argmax(rms_time)
)

show_date = dates[it_show]


finite_eta = etaR[it_show][
    np.isfinite(
        etaR[it_show]
    )
]

vmax = np.percentile(
    np.abs(finite_eta),
    98
)

fig, ax = plt.subplots(
    figsize=(14, 6)
)

p = ax.pcolormesh(
    lon,
    lat,
    etaR[it_show],
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    shading="auto"
)

fig.colorbar(
    p,
    ax=ax,
    label="Rossby-band SSH component [m]"
)


LON2, LAT2 = np.meshgrid(
    lon,
    lat
)

skipx = 5
skipy = 2

q = ax.quiver(
    LON2[::skipy, ::skipx],
    LAT2[::skipy, ::skipx],
    uR[it_show, ::skipy, ::skipx],
    vR[it_show, ::skipy, ::skipx],
    color="black",
    width=0.002
)


# Choose a sensible vector key automatically.
spd = speedR[it_show]
p90 = np.nanpercentile(
    spd,
    90
)

choices = np.array([
    0.02,
    0.05,
    0.10,
    0.20,
    0.50
])

key = choices[
    np.argmin(
        np.abs(
            np.log(
                choices
                /
                max(p90, 1e-4)
            )
        )
    )
]

ax.quiverkey(
    q,
    0.82,
    1.035,
    key,
    f"{key:g} m/s filtered current",
    coordinates="axes",
    labelpos="E"
)

ax.axhline(
    0,
    color="black",
    ls="--",
    lw=0.8
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)

ax.set_title(
    "Rossby-dispersion-band component\n"
    f"NASA SSH + OSCAR water velocity, {show_date.date()}"
)

plt.tight_layout()

plt.savefig(
    "figures/oscar_rossby_filtered_snapshot.png",
    dpi=240
)

plt.close()


# ============================================================
# PLOT 4:
# DOMINANT SPECTRAL-BIN MERIDIONAL STRUCTURE
# ============================================================

eta_prof = (
    FETA[it_peak, :, ik_peak]
)

u_prof = (
    FU[it_peak, :, ik_peak]
)

v_prof = (
    FV[it_peak, :, ik_peak]
)


# Rotate all fields by ONE common phase so that the symmetric
# SSH component is predominantly real and positive.
ref = np.mean(
    eta_prof[
        (np.abs(lat) >= 3)
        &
        (np.abs(lat) <= 8)
    ]
)

phase = np.exp(
    -1j*np.angle(ref)
)

eta_real = np.real(
    eta_prof * phase
)

u_real = np.real(
    u_prof * phase
)

v_real = np.real(
    v_prof * phase
)


# Normalize separately only for SHAPE comparison.
def norm(x):
    m = np.max(
        np.abs(x)
    )

    return (
        x/m
        if m > 0
        else x
    )


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    norm(eta_real),
    lat,
    label="SSH"
)

ax.plot(
    norm(u_real),
    lat,
    label="zonal velocity u"
)

ax.plot(
    norm(v_real),
    lat,
    label="meridional velocity v"
)

ax.axvline(
    0,
    color="black",
    lw=0.7
)

ax.axhline(
    0,
    color="black",
    lw=0.7
)

ax.set_xlim(
    -1.1,
    1.1
)

ax.set_ylim(
    -10,
    10
)

ax.set_xlabel(
    "Normalized modal amplitude"
)

ax.set_ylabel(
    "Latitude"
)

ax.set_title(
    "Meridional structure at dominant Rossby spectral bin\n"
    f"c={c_peak:.2f} m/s, "
    f"T={period_peak_days:.0f} d, "
    f"lambda={wavelength_peak_km:.0f} km"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "figures/oscar_rossby_meridional_mode.png",
    dpi=240
)

plt.close()


# ============================================================
# SAVE FILTERED FIELDS
# ============================================================

out = xr.Dataset(
    data_vars={
        "ssha_rossby": (
            ("time", "lat", "lon"),
            etaR
        ),
        "u_rossby": (
            ("time", "lat", "lon"),
            uR
        ),
        "v_rossby": (
            ("time", "lat", "lon"),
            vR
        ),
        "ug_rossby": (
            ("time", "lat", "lon"),
            ugR
        ),
        "vg_rossby": (
            ("time", "lat", "lon"),
            vgR
        ),
    },
    coords={
        "time": ds.time.values,
        "lat": lat,
        "lon": lon,
    }
)

out.attrs[
    "filter_description"
] = (
    "n=1 equatorial Rossby dispersion-band filter; "
    f"Kelvin reference c={C_K} m/s"
)

out.to_netcdf(
    "data_processed/oscar_rossby_filtered.nc"
)


print()
print("SAVED")
print("-----")
print(
    "figures/oscar_rossby_komega_comparison.png"
)
print(
    "figures/oscar_rossby_parity.png"
)
print(
    "figures/oscar_rossby_filtered_snapshot.png"
)
print(
    "figures/oscar_rossby_meridional_mode.png"
)
print(
    "data_processed/oscar_rossby_filtered.nc"
)
