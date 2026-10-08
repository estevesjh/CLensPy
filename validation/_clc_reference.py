r"""Shared plumbing for the ``validate_cov_*`` ladder: the pinned reference.

The reference is Hao-Yi Wu's ``cluster-lensing-cov`` **at one commit**,
`PIN`. Every rung compares against numbers that commit produced, so the
commit is part of the definition of the reference, not a detail: her
2026-09 commits changed the shape-noise normalisation and the lens-source
cutoff, and a reference that floats with ``master`` would silently change
what "agreement" means.

Locating the code
-----------------
``$CLUSTER_LENSING_COV_DIR`` points at a clone (default
``~/Documents/Dev/github/cluster-lensing-cov``, as for
``validate_lensing_kernel.py``). If that clone's ``HEAD`` is not `PIN`, a
**detached worktree** at `PIN` is created next to the scratch directory
(``$CLC_PINNED_DIR`` overrides the location) -- her working tree is never
checked out, reset or edited.

Only `make_clc_reference.py` (rung V0) imports her code. Every other rung
reads the snapshot it writes, ``validation/data/clc_cddbb2a_reference.npz``,
so the ladder runs without her repository once V0 has been run.

NOTE: **one compatibility shim, applied before her modules load.** Her code
calls ``np.trapz``, which NumPy 2.4 removed; ``np.trapezoid`` is the same
function under its new name (identical composite-trapezoid arithmetic), so
``np.trapz = np.trapezoid`` changes no number. Nothing else is patched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: The upstream commit the reference is pinned to: hywu/cluster-lensing-cov
#: ``cddbb2a`` (2026-09-22, "implement f_src_behind_lens consistently").
PIN = "cddbb2ac7254ae59eb20a7e4e56011a9f661a4ff"

DEFAULT_REPO = Path.home() / "Documents/Dev/github/cluster-lensing-cov"
REPO = Path(os.environ.get("CLUSTER_LENSING_COV_DIR", DEFAULT_REPO))

VALIDATION_DIR = Path(__file__).resolve().parent
DATA_DIR = VALIDATION_DIR / "data"
SNAPSHOT = DATA_DIR / "clc_cddbb2a_reference.npz"
PIN_FILE = DATA_DIR / "cluster_lensing_cov_pin.txt"
FIG_DIR = VALIDATION_DIR.parent / "docs" / "_static" / "validation"

# -- the matched configuration every rung uses --------------------------------
#
# Her DES-Y1 driver (validation/demo_desy1.py) runs h=0.7, OmegaM=0.3,
# OmegaDE=0.7, sigma8=0.8; OmegaB/ns/tau keep her parameters.py defaults.
# The source population is configs/des_y1.json, as in
# validate_lensing_kernel.py. zs_max is pinned to 3.0 on BOTH sides (her
# default became 5 in 964f4b3, ours is 3); V1 reports what the 3 -> 5 change
# does on its own, so it is measured rather than folded into a tolerance.
REF_COSMO = dict(h=0.7, OmegaM=0.3, OmegaDE=0.7, sigma8=0.8, OmegaB=0.045,
                 ns=0.963, tau=0.088, w0=-1.0, wa=0.0)
REF_SOURCES = dict(z_star=0.74, m=1.68, beta=2.33, sigma_gamma=0.3,
                   n_src_arcmin=6.28, zs_min=0.0, zs_max=3.0)

#: One footprint number drives both f_sky and the halo surface density
#: (her DES Y1 driver: 1321 + 116 deg^2).
SURVEY_AREA_DEG2 = 1437.0
FULL_SKY_DEG2 = 41253.0  # her constant; 4 pi (180/pi)^2 = 41252.96
F_SKY = SURVEY_AREA_DEG2 / FULL_SKY_DEG2

#: Her rounded constants (clens/util/constants.py) against clenspy's.
C_HERS, C_OURS = 3.0e5, 299792.458
RHO_CRIT_H2_HERS = 2.775e11  # h^2 Msun/Mpc^3

#: Thin halo slices for V4 (width 0.05: one 0.1-slab each on both sides),
#: and the DES Y1 wide bin for V5. Counts and bias are "precalculated"
#: inputs (her PrecalculatedCountsBias), lambda in [20, 30): the wide-bin
#: counts/bias are McClintock Fig. 4 / her demo_desy1.py; the thin slices
#: take the wide bin's surface density scaled by slice volume.
THIN_SLICES = ((0.25, 0.30), (0.40, 0.45), (0.55, 0.60))
WIDE_BIN = (0.20, 0.35)
WIDE_COUNTS, WIDE_BIAS = 762.0, 2.80019023
SLICE_BIAS = 2.8

#: Comoving, h-free radial edges [Mpc]: geometric, so the FFTLog engine
#: applies, spanning the 0.03-30 Mpc range of the DES Y1 analysis.
RP_EDGES = np.geomspace(0.05, 50.0, 13)


def git(*args, repo=REPO):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def pinned_checkout() -> Path:
    """A directory holding her code at `PIN`, without touching her tree.

    Uses ``$CLUSTER_LENSING_COV_DIR`` directly if its HEAD already is the
    pin; otherwise a detached worktree (``git worktree add --detach``),
    created once and reused.
    """
    override = os.environ.get("CLC_PINNED_DIR")
    if override:
        path = Path(override)
        if path.exists() and git("rev-parse", "HEAD", repo=path) == PIN:
            return path
    else:
        path = Path(tempfile.gettempdir()) / f"clc_{PIN[:7]}"
    if not REPO.exists():
        raise FileNotFoundError(
            f"cluster-lensing-cov not found at {REPO}; set "
            "CLUSTER_LENSING_COV_DIR")
    if git("rev-parse", "HEAD") == PIN:
        return REPO
    try:
        git("cat-file", "-e", PIN + "^{commit}")
    except subprocess.CalledProcessError:
        git("fetch", "origin")  # fetch only: never moves her branches
    if not path.exists():
        git("worktree", "add", "--detach", str(path), PIN)
    if git("rev-parse", "HEAD", repo=path) != PIN:
        raise RuntimeError(f"{path} is not at {PIN}")
    return path


def import_clens():
    """Put the pinned ``clens`` on ``sys.path``, after the np.trapz shim."""
    np.trapz = np.trapezoid  # removed in NumPy 2.4; same function renamed
    root = pinned_checkout()
    for name in list(sys.modules):
        if name == "clens" or name.startswith("clens."):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    import clens  # noqa: F401

    if Path(clens.__file__).resolve().parents[1] != root.resolve():
        raise RuntimeError(f"imported clens from {clens.__file__}, "
                           f"expected {root}")
    return root


def load_snapshot():
    """The V0 snapshot, or a clear instruction to make it."""
    if not SNAPSHOT.exists():
        print(f"snapshot {SNAPSHOT} missing: run "
              "`python validation/make_clc_reference.py` first")
        sys.exit(2)
    snap = dict(np.load(SNAPSHOT, allow_pickle=False))
    pinned = str(snap["pin"])
    if pinned != PIN:
        print(f"snapshot was made at {pinned}, ladder expects {PIN}")
        sys.exit(2)
    return snap


# -- the clenspy side of the matched configuration ----------------------------

def ours_cosmology():
    """Flat LCDM equal to her ``w0waCDM(w0=-1, wa=0)``: Tcmb0 = 0 in both."""
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=100.0 * REF_COSMO["h"], Om0=REF_COSMO["OmegaM"],
                         Tcmb0=0.0)


def ours_survey(zs_max=REF_SOURCES["zs_max"]):
    from clenspy.survey import Survey

    s = REF_SOURCES
    return Survey.smail(z_star=s["z_star"], m=s["m"], beta=s["beta"],
                        sigma_gamma=s["sigma_gamma"],
                        n_src_arcmin=s["n_src_arcmin"], zs_min=s["zs_min"],
                        zs_max=zs_max)


def pk_from_snapshot(snap):
    r"""Her linear :math:`P(k, z)` [Mpc^3, k in 1/Mpc] as a callable.

    Stored as :math:`P(k, 0)` on a dense log grid times :math:`D^2(z)`
    on a z grid -- exact for her Eisenstein-Hu spectrum (no neutrinos, so
    growth is scale-independent; V0 asserts that to 1e-10 before storing).
    Interpolation error is measured in V0 and recorded as ``pk_interp_err``.
    """
    from scipy.interpolate import CubicSpline

    lnk, lnp0 = np.log(snap["pk_k"]), np.log(snap["pk_p0"])
    spl_p = CubicSpline(lnk, lnp0)
    spl_d2 = CubicSpline(snap["pk_z"], snap["pk_d2"])

    def pk(k, z):
        k = np.asarray(k, dtype=float)
        return np.exp(spl_p(np.log(k))) * float(spl_d2(z))

    return pk


def her_rho_mean():
    """Her comoving mean density, with her rounded rho_crit."""
    return RHO_CRIT_H2_HERS * REF_COSMO["h"] ** 2 * REF_COSMO["OmegaM"]


class Verdicts:
    """Collect PASS/FAIL/INFO rows; print a table; set the exit status."""

    def __init__(self, rung):
        self.rung = rung
        self.rows = []

    def check(self, name, value, tol, *, kind="rel", note=""):
        ok = bool(np.isfinite(value) and abs(value) <= tol)
        self.rows.append((name, value, tol, "PASS" if ok else "FAIL", note))
        return ok

    def expect_fail(self, name, value, tol, note=""):
        """A comparison that documents a known disagreement (a found bug).

        Recorded as ``FAIL*`` -- a failure with a diagnosis attached in
        ``validation/covariance_review_REPORT.md``. It still makes the rung
        exit nonzero: the code as shipped disagrees with the reference, and
        a known cause is not a pass. It turns into PASS once src is fixed.
        """
        ok = bool(np.isfinite(value) and abs(value) <= tol)
        self.rows.append((name, value, tol, "PASS" if ok else "FAIL*", note))
        return ok

    def info(self, name, value, note=""):
        self.rows.append((name, value, np.nan, "INFO", note))

    def report(self):
        print(f"\n== {self.rung} ==")
        print(f"{'check':<58s} {'value':>11s} {'tol':>9s}  verdict")
        for name, value, tol, verdict, note in self.rows:
            t = "" if not np.isfinite(tol) else f"{tol:9.2e}"
            print(f"{name:<58s} {value:11.4e} {t:>9s}  {verdict}"
                  + (f"   # {note}" if note else ""))
        failed = [r for r in self.rows if r[3] == "FAIL"]
        known = [r for r in self.rows if r[3] == "FAIL*"]
        print(f"-> {self.rung}: {len(failed)} unexpected FAIL, "
              f"{len(known)} known-bug FAIL*, "
              f"{sum(r[3] == 'PASS' for r in self.rows)} PASS")
        return 1 if (failed or known) else 0
