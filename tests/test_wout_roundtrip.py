from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from vmec_jax.wout import read_wout, write_wout
pytestmark = pytest.mark.full


_CASES = [
    "examples/data/wout_circular_tokamak_reference.nc",
    "examples/data/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
]


@pytest.mark.parametrize("wout_rel", _CASES)
def test_wout_roundtrip_read_write_read(tmp_path: Path, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    src = root / wout_rel
    assert src.exists()

    w0 = read_wout(src)
    out = tmp_path / Path(wout_rel).name
    write_wout(out, w0, overwrite=True)
    w1 = read_wout(out)

    # Compare all fields that we read/write.
    assert w0.ns == w1.ns
    assert w0.mpol == w1.mpol
    assert w0.ntor == w1.ntor
    assert w0.nfp == w1.nfp
    assert w0.lasym == w1.lasym
    assert w0.signgs == w1.signgs

    for name in [
        "xm",
        "xn",
        "xm_nyq",
        "xn_nyq",
    ]:
        assert np.array_equal(getattr(w0, name), getattr(w1, name))

    for name in [
        "rmnc",
        "rmns",
        "zmnc",
        "zmns",
        "lmnc",
        "lmns",
        "phipf",
        "chipf",
        "phips",
        "q_factor",
        "chi",
        "mass",
        "beta_vol",
        "over_r",
        "gmnc",
        "gmns",
        "bsupumnc",
        "bsupumns",
        "bsupvmnc",
        "bsupvmns",
        "bsubumnc",
        "bsubumns",
        "bsubvmnc",
        "bsubvmns",
        "currumnc",
        "currvmnc",
        "bmnc",
        "bmns",
        "vp",
        "pres",
        "presf",
        "fsqt",
        "wdot",
        "am",
        "ai",
        "am_aux_s",
        "am_aux_f",
        "ac_aux_s",
        "ac_aux_f",
        "ai_aux_s",
        "ai_aux_f",
    ]:
        a = np.asarray(getattr(w0, name))
        b = np.asarray(getattr(w1, name))
        assert a.shape == b.shape
        assert np.allclose(a, b, rtol=0.0, atol=0.0)

    for name in [
        "wb",
        "volume_p",
        "version_",
        "gamma",
        "wp",
        "fsqr",
        "fsqz",
        "fsql",
        "ftolv",
        "aspect",
        "betatotal",
        "betapol",
        "betator",
        "betaxis",
        "rmax_surf",
        "rmin_surf",
        "zmax_surf",
        "rbtor0",
        "rbtor",
        "IonLarmor",
        "volavgB",
    ]:
        assert float(getattr(w0, name)) == float(getattr(w1, name))

    for name in ["niter", "itfsq", "ier_flag"]:
        assert int(getattr(w0, name)) == int(getattr(w1, name))

    for name in ["lrecon", "lfreeb", "lrfp"]:
        assert bool(getattr(w0, name)) is bool(getattr(w1, name))

    for name in ["input_extension", "mgrid_file", "pmass_type", "pcurr_type", "piota_type"]:
        assert str(getattr(w0, name)) == str(getattr(w1, name))
