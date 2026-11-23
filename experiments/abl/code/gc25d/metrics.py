"""视频指标：逐帧 / 逐 clip 的 PSNR、SSIM（生成 vs GT），以及分割一致性 mIoU 的预留接口。

约定
--------------------
* 输入一律 uint8 RGB，[F, H, W, 3] 或单帧 [H, W, 3]，GT 与生成必须同形状（不做任何隐式缩放）。
* PSNR：MAX = 255，逐帧 10·log10(255² / MSE)，MSE 在 H·W·3 上取均值；完全相同的帧返回 inf
  （汇总时 inf 不进均值，单独计数，见 clip_metrics）。
* SSIM：Wang et al. 2004 的标准设定，与 skimage.metrics.structural_similarity(
  channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) 同口径：
  11×11 高斯窗（σ=1.5，截断 3.5σ ⇒ 半径 5），K1=0.01，K2=0.03，逐通道算 SSIM 图、裁掉 5 px 边框后取均值，
  再对 3 个通道取均值。环境里没有 skimage，这里用 cv2.GaussianBlur 自己实现；tests/test_infer.py 用
  scipy.ndimage.gaussian_filter 写的独立参考实现逐位核对（差 < 1e-6）。
* clip 汇总：帧 0 是条件帧（VAE 编码再解码的首帧，不是生成的），单独报 *_frame0；主指标 *_mean 只对帧 1..F-1
  取均值；另给 psnr_of_mean_mse（先对 1..F-1 的 MSE 取均值再换算，不受单帧 inf/极值影响）。

分割一致性 mIoU（预留，尚未实现）
-------------------------------
segmentation_consistency() 是给"生成画面是否服从数值语义图"的指标留的钩子：将来接一个 RGB → 类别 id 的分割器
（例如在本数据 GT 上训的轻量 segformer），对生成帧做分割，与条件 ids 比 mIoU。分割器没接上之前它返回 None，
eval 的 json 里就是 null —— 不会伪造数字。miou() 本身（给定两张 id 图）已实现并有测试。
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import cv2
import numpy as np

SSIM_SIGMA = 1.5
SSIM_RADIUS = 5            # int(3.5 * 1.5 + 0.5)，与 scipy/skimage 的 truncate=3.5 一致
SSIM_K1, SSIM_K2 = 0.01, 0.03
DATA_RANGE = 255.0


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    assert a.dtype == np.uint8 and b.dtype == np.uint8, f"指标输入必须是 uint8，得到 {a.dtype} / {b.dtype}"
    assert a.shape == b.shape, f"生成与 GT 形状不同：{a.shape} vs {b.shape}"
    assert a.ndim in (3, 4) and a.shape[-1] == 3, f"应为 [H,W,3] 或 [F,H,W,3]，得到 {a.shape}"


# ------------------------------------------------------------------ PSNR ----
def mse_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[F,H,W,3] × 2 → float64 [F]（单帧输入返回 [1]）。"""
    _check_pair(a, b)
    if a.ndim == 3:
        a, b = a[None], b[None]
    out = np.empty(a.shape[0], np.float64)
    for i in range(a.shape[0]):           # 逐帧，避免 81×480×832×3 的 float64 大临时数组
        d = a[i].astype(np.float32) - b[i].astype(np.float32)
        out[i] = float(np.mean(d.astype(np.float64) ** 2))
    return out


def psnr_from_mse(mse) -> np.ndarray:
    mse = np.asarray(mse, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(mse == 0, np.inf, 10.0 * np.log10(DATA_RANGE ** 2 / np.maximum(mse, 1e-300)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """单帧 PSNR（dB）。"""
    assert a.ndim == 3, f"psnr() 只收单帧 [H,W,3]，多帧用 psnr_per_frame；得到 {a.shape}"
    return float(psnr_from_mse(mse_per_frame(a, b))[0])


def psnr_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return psnr_from_mse(mse_per_frame(a, b))


# ------------------------------------------------------------------ SSIM ----
def _blur(x: np.ndarray) -> np.ndarray:
    k = 2 * SSIM_RADIUS + 1
    return cv2.GaussianBlur(x, (k, k), SSIM_SIGMA, borderType=cv2.BORDER_REFLECT)


def ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    """单通道 SSIM（float64 [H,W]，值域 0..255）。总体协方差（use_sample_covariance=False），裁 5 px 边框后取均值。"""
    assert x.shape == y.shape and x.ndim == 2, (x.shape, y.shape)
    c1 = (SSIM_K1 * DATA_RANGE) ** 2
    c2 = (SSIM_K2 * DATA_RANGE) ** 2
    mx, my = _blur(x), _blur(y)
    sxx = _blur(x * x) - mx * mx
    syy = _blur(y * y) - my * my
    sxy = _blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    r = SSIM_RADIUS
    return float(s[r:-r, r:-r].mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """单帧 RGB SSIM：3 个通道各算一次取均值。"""
    _check_pair(a, b)
    assert a.ndim == 3, f"ssim() 只收单帧 [H,W,3]，多帧用 ssim_per_frame；得到 {a.shape}"
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean([ssim_channel(a64[..., c], b64[..., c]) for c in range(3)]))


def ssim_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _check_pair(a, b)
    if a.ndim == 3:
        a, b = a[None], b[None]
    return np.array([ssim(a[i], b[i]) for i in range(a.shape[0])], np.float64)


# ------------------------------------------------------------------ clip 汇总 ----
def _finite_mean(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def clip_metrics(gen: np.ndarray, gt: np.ndarray, with_ssim: bool = True) -> dict:
    """一条 clip 的全部指标。gen / gt：uint8 [F,H,W,3]（F=81）。

    返回 dict（全是 python float / list，可直接 json.dump）：
      psnr / ssim                  逐帧列表 [F]（inf 写成 None）
      psnr_frame0 / ssim_frame0    条件帧（只反映 VAE 往返误差）
      psnr_mean / ssim_mean        帧 1..F-1 的均值（主指标；inf 不计入，个数见 psnr_n_inf）
      psnr_of_mean_mse             帧 1..F-1 的 MSE 均值换算的 PSNR
      psnr_last / ssim_last        最后一帧（误差随时间累积，看末端）
    """
    _check_pair(gen, gt)
    assert gen.ndim == 4 and gen.shape[0] >= 2, f"clip 至少 2 帧，得到 {gen.shape}"
    mse = mse_per_frame(gen, gt)
    p = psnr_from_mse(mse)
    out = {
        "n_frames": int(gen.shape[0]),
        "psnr": [None if not math.isfinite(v) else float(v) for v in p],
        "psnr_frame0": None if not math.isfinite(p[0]) else float(p[0]),
        "psnr_mean": _finite_mean(p[1:]),
        "psnr_n_inf": int(np.sum(~np.isfinite(p[1:]))),
        "psnr_of_mean_mse": float(psnr_from_mse(mse[1:].mean())),
        "psnr_last": None if not math.isfinite(p[-1]) else float(p[-1]),
    }
    if with_ssim:
        s = ssim_per_frame(gen, gt)
        out.update(ssim=[float(v) for v in s], ssim_frame0=float(s[0]), ssim_mean=float(s[1:].mean()),
                   ssim_last=float(s[-1]))
    return out


def aggregate(per_clip: list[dict]) -> dict:
    """多条 clip 的汇总：标量取均值 / 中位数；逐帧曲线逐帧取均值（各 clip 帧数必须相同）。"""
    assert per_clip, "没有 clip 可汇总"
    out: dict = {"n_clips": len(per_clip)}
    for key in ("psnr_mean", "psnr_of_mean_mse", "psnr_frame0", "psnr_last", "ssim_mean", "ssim_frame0", "ssim_last"):
        vals = [c[key] for c in per_clip if c.get(key) is not None]
        if vals:
            out[key] = float(np.mean(vals))
            out[key + "_median"] = float(np.median(vals))
            out[key + "_std"] = float(np.std(vals))
    for key in ("psnr", "ssim"):
        curves = [c[key] for c in per_clip if c.get(key) is not None]
        if curves:
            n = {len(c) for c in curves}
            assert len(n) == 1, f"{key} 曲线长度不一：{n}"
            arr = np.array([[np.nan if v is None else v for v in c] for c in curves], np.float64)
            ok = np.isfinite(arr)
            n_ok = ok.sum(0)
            mean = np.where(ok, arr, 0.0).sum(0) / np.maximum(n_ok, 1)
            out[key + "_curve"] = [float(m) if k > 0 else None for m, k in zip(mean, n_ok)]
    return out


# ------------------------------------------------------------------ 分割一致性（预留） ----
# 允许的条件类别；255 = UNKNOWN，评测时忽略
SEM_CLASSES = (0, 1, 2, 3, 4, 5, 6, 10, 11)
UNKNOWN_ID = 255


def miou(pred: np.ndarray, gt: np.ndarray, classes=SEM_CLASSES, ignore: int = UNKNOWN_ID) -> dict:
    """两张（或两叠）id 图的 mIoU。gt 里 == ignore 的像素不计；某类在 pred ∪ gt 都没出现 ⇒ 该类不进均值。
    返回 {"miou": float|None, "iou": {cls: float|None}}。"""
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    valid = gt != ignore
    iou = {}
    for c in classes:
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = int(np.count_nonzero(p | g))
        iou[int(c)] = None if union == 0 else float(np.count_nonzero(p & g)) / union
    vals = [v for v in iou.values() if v is not None]
    return {"miou": float(np.mean(vals)) if vals else None, "iou": iou}


# 分割器签名：uint8 RGB [F,H,W,3] → uint8 id [F,H,W]（类别取值同 SEM_CLASSES）
Segmenter = Callable[[np.ndarray], np.ndarray]


def segmentation_consistency(gen: np.ndarray, sem_ids: np.ndarray, segmenter: Optional[Segmenter] = None) -> Optional[dict]:
    """==== HOOK：分割一致性 mIoU（未来指标）====
    思路：segmenter(生成帧) → 预测 id 图，与条件 sem_ids [F,H,W] 比 mIoU（UNKNOWN=255 区域不计），
    衡量"改了数字 ⇒ 画面那里真的变成那一类"。帧 0 是条件帧，只用帧 1..F-1。
    segmenter=None（现状：还没有训好的 RGB→id 分割器）⇒ 返回 None，不出数字。
    接入时只需传入 segmenter，infer.py 的 eval 会把结果写进 metrics json 的 "seg_consistency" 字段。"""
    if segmenter is None:
        return None
    assert gen.ndim == 4 and sem_ids.shape == gen.shape[:3], (gen.shape, sem_ids.shape)
    pred = segmenter(gen[1:])
    assert pred.shape == sem_ids[1:].shape and pred.dtype == np.uint8, (pred.shape, pred.dtype)
    return miou(pred, sem_ids[1:])
