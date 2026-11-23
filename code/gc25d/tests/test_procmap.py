"""procmap：CPU 上的求交 / 地面纹理小测（真实 clip 的整套生成见 procmap.py --default4 的 report.json）。"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import procmap as PM


def _prims():
    return [PM.box([0, 0], 200, 100, 150, 0.3, 0, 3), PM.ell([300, 0, 50], [80, 60, 40], 0.7, 5),
            PM.cyl([0, 300], 40, 0, 220, 6), PM.gable([0, -300], 400, 200, 0.2, 300, 100, 30, 3),
            PM.prism(PM._ngon([600, 600], 100, 8), 0, 250, 4, taper=0.4), PM.log(type("W", (), dict(
                rng=np.random.default_rng(0), z0=0.0))(), np.array([-400.0, 400.0]))[0][0]]


def test_cast_matches_sdf_marching():
    prims = _prims()
    rng = np.random.default_rng(0)
    ro = rng.uniform(-700, 700, (300, 3)); ro[:, 2] = 900
    rd = rng.normal(0, 1, (300, 3)); rd[:, 2] = -np.abs(rd[:, 2]) - 0.5
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    t, k = PM.Caster(prims, "cpu").cast(ro, rd)
    ts = np.linspace(0, 3000, 30001)
    for i in range(len(ro)):
        P = ro[i] + ts[:, None] * rd[i]
        s = np.min([PM.sdf_lower(p, P) for p in prims], 0)
        tr = ts[np.argmax(s <= 0)] if (s <= 0).any() else np.inf
        assert (np.isinf(tr) and k[i] == -1) or abs(tr - t[i]) < 0.2, (i, tr, t[i])


def test_ground_layers_paint_in_order():
    g = PM.Ground()
    g.layers.append(("seg", PM.PAVING, np.array([[0, 0, 1000, 0.0]]), np.array([100.0])))
    g.layers.append(("disk", PM.TERRAIN, np.array([500, 0, 50.0])))
    g.layers.append(("poly", PM.PAVING, PM._rect([0, 500], 200, 200, 0.0)))
    xy = np.array([[200, 50], [200, 150], [500, 0], [0, 550], [0, 650]], float)
    assert PM.Ground.classify(g, xy, "cpu").tolist() == [2, 1, 1, 2, 1]
