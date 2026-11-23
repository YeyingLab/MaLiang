#!/usr/bin/env python3
"""生成器 EMA：fp32 影子权重（stage2 CD 的目标网络 / stage2·stage3 的部署权重）。

    from distill.ema import EmaShadow, ema_update
    ema = EmaShadow(build_arope_dit(...), device="cuda")      # 不进 accelerator.prepare
    ...
    ema.update(student, decay=(0.0 if step < args.ema_start else args.ema_decay))
    runlib.save_dit(accel, ema, runlib.ema_path(run_dir, step), {"step": step, "ema_decay": decay})

★ 影子必须是 fp32。bf16 只有 8 位尾数，相对精度 ~2^-8 ≈ 0.4%：decay=0.99 时
`pe = 0.99*pe + 0.01*p`，那 1% 的增量落在 pe 尾数之下会被舍入直接吃掉 —— 影子权重在数值上
纹丝不动，loss 却一切正常（CD 的 loss 本来就从 0 开始涨）。这不是精度损失，是目标网络失效。
tests/test_runlib.py::test_bf16_shadow_drops_the_update 把这条钉成了断言。
fp32 影子只是存储；前向照样在 autocast(bf16) 下跑，算力开销不变（多 20 GB 显存，见 PORT_设计 预算表）。

★ EmaShadow 故意**不是** nn.Module。accelerate 的 prepare 会 `for obj in result: if isinstance(obj, nn.Module): model = obj`
（accelerator.py:2229-2236）—— 最后一个 nn.Module 静默胜出，其余一个警告都没有就没被包装。
影子不是 nn.Module 就绝不可能被误传进 prepare、被卷进优化器的参数表、或者混进上层模块的 state_dict。
它照样能当模型调用（__call__ 转发给内部 module）。

stage3 的 EMA 是 FSDP 包装的 AropeDiT（要按分片 zip），不走 EmaShadow，直接用模块级的
ema_update(ema, student, decay)；两条路的数学完全一样。
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _bare(model: nn.Module) -> nn.Module:
    """剥掉 DDP / DeepSpeedEngine / EmaShadow 的外壳。"""
    seen = 0
    while hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module) and seen < 8:
        model = model.module
        seen += 1
    return model


@torch.no_grad()
def ema_update(ema: nn.Module, student, decay: float) -> None:
    """逐参数 EMA：`pe = decay*pe + (1-decay)*p`，decay <= 0 ⇒ 硬拷贝。

    zip 成立的前提：两边暴露的是**同一组、同一顺序、同一分片方式**的 Parameter。
    ZeRO-2 不分片参数，天然成立；FSDP 要 fsdp_use_orig_params=true（PORT_设计 R5）。
    这里额外核对个数与逐个形状，顺序错位在绝大多数情况下会当场炸出来，而不是静默训歪。
    student 可以是 DeepSpeedEngine / DDP —— nn.Module 外壳的 parameters() 顺序与内部一致。"""
    pe_list = list(ema.parameters())
    ps_list = list(student.parameters())
    if len(pe_list) != len(ps_list):
        raise RuntimeError(f"EMA 参数个数不匹配：影子 {len(pe_list)} vs 学生 {len(ps_list)}")
    for i, (pe, p) in enumerate(zip(pe_list, ps_list)):
        if pe.shape != p.shape:
            raise RuntimeError(f"EMA 第 {i} 个参数形状不匹配：影子 {tuple(pe.shape)} vs 学生 {tuple(p.shape)}")
        src = p.data
        if src.device != pe.device:
            src = src.to(pe.device, non_blocking=True)
        if decay <= 0.0:
            pe.data.copy_(src)                    # 硬拷贝（step < ema_start）
        else:
            # dtype 由 TensorIterator 就地提升：src 是 bf16 也不会额外分配一份 fp32
            pe.data.mul_(decay).add_(src, alpha=1.0 - decay)


class EmaShadow:
    """fp32 影子权重。不是 nn.Module（见文件头），但能当模型直接调用。

    EmaShadow(module)               —— deepcopy 一份，源模型归调用方（多占一份显存/内存）
    EmaShadow(module, own=True)     —— 直接接管这个 module（stage2 已经单独建了第三个模型时用这个，
                                       不再多复制一份 5B 权重）
    device="cpu" 可以把影子放在主存里（ReactiveGWM 的 *_ema_shadow_device: cpu），
    update 会自动把学生的分片搬过来；但前向就得先 .to("cuda")，stage2 默认放卡上。
    """

    def __init__(self, module: nn.Module, *, device=None, own: bool = False, dtype: torch.dtype = torch.float32):
        if dtype != torch.float32:
            raise ValueError("影子必须是 fp32：bf16 在 decay=0.99 下会把 1% 的增量整个舍掉（见文件头）")
        src = _bare(module)
        self.module = src if own else copy.deepcopy(src)
        self.module.to(dtype=dtype)
        if device is not None:
            self.module.to(device=device)
        self.module.requires_grad_(False).eval()
        self.dtype = dtype
        self._assert_fp32()

    # ---- 数值 ----
    def _assert_fp32(self) -> None:
        bad = [n for n, p in self.module.named_parameters() if p.dtype is not torch.float32]
        if bad:
            raise RuntimeError(f"影子权重不是 fp32：{bad[:5]}…({len(bad)})")

    @torch.no_grad()
    def update(self, student, decay: float) -> None:
        """一个 optimizer step 调一次（不是每个 micro-batch）。decay <= 0 ⇒ 硬拷贝。"""
        ema_update(self.module, student, decay)

    @torch.no_grad()
    def hard_copy(self, student) -> None:
        """影子 ← 学生。step < ema_start 期间每步都调，等价于 update(student, 0.0)。"""
        ema_update(self.module, student, 0.0)

    # ---- 当模型用 ----
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def buffers(self, recurse: bool = True):
        return self.module.buffers(recurse=recurse)

    def to(self, *args, **kwargs):
        """只搬设备。dtype 不许动：.to(torch.bfloat16) 会当场毁掉目标网络。

        注意 dit.freqs 是普通的 tuple 属性（不是 buffer），Module.to 搬不动它 ——
        影子搬完设备后调用方要照 train.py:461 那样补一句
        `ema.module.dit.freqs = tuple(t.to(device) for t in ema.module.dit.freqs)`。"""
        if any(isinstance(a, torch.dtype) for a in args) or kwargs.get("dtype") is not None:
            raise RuntimeError("影子权重的 dtype 不许改（fp32），只搬设备")
        self.module.to(*args, **kwargs)
        return self

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode: bool = True):
        """影子永远是 eval —— 显式挡住 train()，免得被跟着学生一起 .train() 打开 dropout。"""
        if mode:
            raise RuntimeError("EMA 影子不训练，别对它调 train()")
        return self.eval()

    # ---- 导出 ----
    def state_dict(self, *args, **kwargs):
        """直接透传内部 module 的键集（不带 "module." 前缀），
        这样 runlib.save_dit(accel, ema, ...) 剥掉 `dit.` 之后得到的就是标准的 827 键。"""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        out = self.module.load_state_dict(*args, **kwargs)
        self._assert_fp32()
        return out

    def export_state_dict(self, dtype: torch.dtype = torch.bfloat16) -> dict:
        """给不走 runlib.save_dit 的场合用：CPU + contiguous + 指定 dtype。"""
        return {k: v.detach().to(device="cpu", dtype=dtype).contiguous()
                for k, v in self.module.state_dict().items()}

    def __repr__(self):
        n = sum(p.numel() for p in self.module.parameters())
        dev = next(self.module.parameters()).device if n else "?"
        return f"EmaShadow({type(self.module).__name__}, {n:,} params, fp32, {dev})"
