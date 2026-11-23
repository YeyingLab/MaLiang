# Release manifest — MaLiang

This release contains the code and the five checkpoints behind the paper's reported results.

## Code layout

| Path | Content |
| --- | --- |
| `code/gc25d/` | model, SFT, inference, preprocessing, `gcdistill` (three-stage distillation) |
| `code/arope/` | ARoPE modules (Python only) |
| `experiments/` | evaluation, edit, unseen-map and ablation scripts, 100-clip test manifests |
| `scripts/`, `configs/` | launchers and accelerate configurations |

Internal paths, credentials, result images/videos, and raw data are not included.

## Checkpoints

All files are bf16 safetensors, `vocab=parts_fx`, 832×480.

| File | SHA-256 | Role |
| --- | --- | --- |
| `sft_fxcond_v1_step-10667.safetensors` | `e476ef1c496439723b10be7edec470540bd60b51960f0ee6ca940037a94306e2` | bidirectional teacher, 831 keys, 10,667 steps |
| `fxcond_v1_stage3_ema_step500.safetensors` | `8d34f58c43f42866f85f9fed6425a428de324e82d2a4d04075ec3080caac0c5f` | deployed causal student, stage-3 DMD EMA 500 |
| `abl_color_vae_step-5000.safetensors` | `d3884c9fa3fcb92d1e932e8130ad26b9e35103a724e4b93f3e94b3ad2c541331` | ablation ①, 827 keys, `color_vae` |
| `abl_color_light_step-5000.safetensors` | `bda67d06d8ba32d11c856399cc9ab64d27e314082208aed5e48c69bfdb5056b8` | ablation ②, `color_light` |
| `abl_v3_ids_step-5000.safetensors` | `a0d4b5c2393dba8ab5c2b5c879b25b1fff502b540467232982a990896bdce680` | ablation ③/④, integer ids |

`abl_v3_ids_step-5000` is the step-5000 snapshot of the same SFT run as the teacher; it uses `parts_fx`, 831 keys.

## Assets

| File | Purpose | Source |
| --- | --- | --- |
| `assets/text_table_gc25d.pt` | precomputed UMT5 text embeddings | project data |
| `assets/taew2_2_super.pth` | TAEHV decoder for the rollout evaluator | TAEHV |
| `assets/i3d_torchscript.pt` | I3D features for FVD | StyleGAN-V lineage |
| `assets/seg_b2_final/` | SegFormer-B2 segmenter fine-tuned on recorded RGB↔ids | project |

## What is not included

- Raw recordings and the Wan2.2-TI2V-5B base model (download separately).
- Logs, result videos, and credentials.
- The baseline models (SolarRenderer, AnyGameRenderer, CWM) and the FLUX first-frame generator used by some experiments.

## Verification

- `models.sha256` — checksums of the five checkpoints.
- `SHA256SUMS` — checksums of every tracked source/config/manifest file.
- `python -m py_compile` passes for all Python files; `bash -n` passes for all shell
  scripts.

## Licensing

Apache-2.0 for this repository. See `LICENSE` and `NOTICE` for third-party components.
