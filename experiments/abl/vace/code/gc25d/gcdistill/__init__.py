"""gc25d 的三阶段蒸馏（Causal Forcing++：stage1 teacher forcing 因果化 → stage2 因果一致性蒸馏 → stage3 self-forcing DMD）。

建在 ARoPE/distill（只 import，不改那边任何文件）之上：
  * 因果掩码 / TF 双流 / 滑窗 KV / FSDP 入口 / 调度器 / EMA / checkpoint 约定 —— 全部直接用 ARoPE/distill 的；
  * gc25d 只多一个逐 cell 的语义图条件（gc_sem_cond），接在 AropeDiT 的掩码通道那个入口上（gc_dit.GcDiT）。
包名不叫 distill：PYTHONPATH 是 gc25d:ARoPE，同名包会把 ARoPE/distill 遮住。
"""
