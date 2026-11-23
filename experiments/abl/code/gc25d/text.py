"""gc25d 逐 cell 动作文本。

模板：  "In a 2.5D isometric view, the player is {motion}{atk}."
    motion 由该 0.25 s block 的按键决定（屏幕相对：W 上、A 左、S 下、D 右；相反键互相抵消）：
        standing still | moving up | moving down | moving left | moving right |
        moving up-left | moving up-right | moving down-left | moving down-right
    atk = ", attacking"（attack==1）否则 ""。共 9 × 2 = 18 条。
空条件串（文本 dropout / CFG）：NULL_PROMPT = "In a 2.5D isometric view."

时间对齐：cell j（1..20）← block s/4 + j − 1；cell 0 是干净首帧，沿用 cell 1 的 block。
"""
from __future__ import annotations

import numpy as np

TEMPLATE = "In a 2.5D isometric view, the player is {motion}{atk}."
NULL_PROMPT = "In a 2.5D isometric view."
ATTACK_SUFFIX = ", attacking"
CELLS = 21
BLOCKS_PER_CLIP = 20
ACTION_KEYS = ("w", "a", "s", "d", "attack")

# (up, right) ∈ {−1,0,1}² → motion；up = W − S，right = D − A（相反键抵消）。顺序即 设计 列表顺序。
MOTIONS = {
    (0, 0): "standing still",
    (1, 0): "moving up",
    (-1, 0): "moving down",
    (0, -1): "moving left",
    (0, 1): "moving right",
    (1, -1): "moving up-left",
    (1, 1): "moving up-right",
    (-1, -1): "moving down-left",
    (-1, 1): "moving down-right",
}


def _bit(name: str, v) -> int:
    iv = int(v)
    if iv not in (0, 1) or iv != v:
        raise ValueError(f"按键 {name} 必须是 0/1，得到 {v!r}")
    return iv


def block_to_prompt(w, a, s, d, attack) -> str:
    """一个 0.25 s block 的 [w, a, s, d, attack] → prompt。"""
    w, a, s, d, attack = (_bit(k, v) for k, v in zip(ACTION_KEYS, (w, a, s, d, attack)))
    motion = MOTIONS[(w - s, d - a)]
    return TEMPLATE.format(motion=motion, atk=ATTACK_SUFFIX if attack else "")


def all_prompts() -> list[str]:
    """全部 18 条动作 prompt（固定顺序：9 个 motion × {不攻击, 攻击}），不含 NULL_PROMPT。"""
    out = [TEMPLATE.format(motion=m, atk=atk) for atk in ("", ATTACK_SUFFIX) for m in MOTIONS.values()]
    assert len(out) == 18 and len(set(out)) == 18
    return out


def cell_blocks(blocks20: np.ndarray) -> np.ndarray:
    """uint8 [20, 5]（block s/4 … s/4+19）→ uint8 [21, 5] 的 cell 顺序（cell 0 = cell 1）。"""
    b = np.asarray(blocks20)
    if b.shape != (BLOCKS_PER_CLIP, 5):
        raise ValueError(f"blocks20 形状应为 [20, 5]，得到 {b.shape}")
    if not np.isin(b, (0, 1)).all():
        raise ValueError(f"blocks20 只能含 0/1，得到 {np.unique(b)}")
    return np.concatenate([b[:1], b], 0).astype(np.uint8)


def cell_prompts(blocks20: np.ndarray) -> list[str]:
    """uint8 [20, 5] → 21 条逐 cell prompt（cell 0 = cell 1）。"""
    cb = cell_blocks(blocks20)
    out = [block_to_prompt(*row.tolist()) for row in cb]
    assert len(out) == CELLS
    return out
