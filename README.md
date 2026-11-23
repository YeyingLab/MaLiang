# MaLiang: Painting Playable Game Worlds from Numbers

Code and checkpoints for **“MaLiang: Painting Playable Game Worlds from Numbers.”**

MaLiang renders a game world directly from the game state. The state is rasterised into a
per-pixel `uint8` **integer class map** and injected into a 5B-parameter Wan2.2-TI2V video
diffusion transformer through a small class-pooling encoder — no rendered proxy video and
no video-VAE encode of the state. A three-stage distillation pipeline (teacher forcing →
consistency distillation → DMD) turns the bidirectional teacher into a causal four-step
student for real-time streaming.

This repository contains the code that produced the paper's numbers and the five checkpoints used
for the reported tables.

## Model zoo

All checkpoints are bf16 safetensors, `vocab=parts_fx`, resolution 832×480. Download the
model archive and put the files under `models/`.

| File | Role | Metadata |
| --- | --- | --- |
| `sft_fxcond_v1_step-10667.safetensors` | Bidirectional teacher | 831 keys, 10,667 SFT steps; teacher referenced by the deploy checkpoint |
| `fxcond_v1_stage3_ema_step500.safetensors` | Deployed causal student | 831 keys, stage-3 DMD EMA 500, 4-step; used for the paper's real-time table |
| `abl_color_vae_step-5000.safetensors` | Ablation ①: coloured video → Wan VAE | 827 keys, `cond_mode=color_vae`, step 5000 |
| `abl_color_light_step-5000.safetensors` | Ablation ②: coloured video → light encoder | 831 keys, `cond_mode=color_light`, step 5000 |
| `abl_v3_ids_step-5000.safetensors` | Ablation ③/④: integer map → `gc_sem_cond` | 831 keys, step 5000; step-5000 snapshot of the same SFT run as the teacher |

`models.sha256` in the model archive lists the SHA-256 of each released file.

## Repository layout

```
maliang/
├── code/
│   ├── gc25d/                 # model, SFT, inference, preprocessing
│   │   ├── model.py           # SemCondEncoder + gc_forward
│   │   ├── train.py           # bidirectional SFT
│   │   ├── infer.py           # clip / user / batch inference
│   │   ├── sample.py          # teacher evaluation on val clips
│   │   ├── sample_user.py     # own first frame + ids + prompts
│   │   ├── prep_v10.py        # raw recordings -> latents / cond / meta
│   │   ├── procmap.py         # procedural unseen maps
│   │   └── gcdistill/         # TF -> CD -> DMD, rollout eval
│   ├── arope/                 # ARoPE modules (vendored, read-only)
│   └── taehv/                 # TAEHV decoder used by rollout eval
├── configs/                   # accelerate / DeepSpeed configurations
├── scripts/train_sft.sh       # SFT launcher
├── experiments/
│   ├── metrics/               # common evaluation + bootstrap CI
│   ├── exp4/                  # edit while playing
│   ├── exp6/                  # unseen maps / FX controls
│   ├── abl/                   # interface ablation code + launchers
│   └── testset/               # 100-clip manifests
├── assets/                    # text table, TAE, I3D, segmenter
├── models/                    # checkpoints (download)
├── requirements.txt
├── LICENSE
└── NOTICE
```

## Installation

Python 3.12, PyTorch 2.11 with CUDA 12.8, and DiffSynth-Studio v2.1.6. The code uses SDPA
only; do **not** install `flash-attn` or `sageattention` (the model asserts they are absent
so the reference path stays bit-exact).

```bash
python3.12 -m venv .venv
. .venv/bin/activate

pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# DiffSynth-Studio (the paper used commit 4dbf980d4d0eb34eda136300dd0d72014cff8965)
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
cd third_party/DiffSynth-Studio
git checkout 4dbf980d4d0eb34eda136300dd0d72014cff8965
pip install -e . --no-deps
cd ../..
```

For training, install `deepspeed==0.19.6` (in `requirements.txt`).

The launchers use `$PWD/.venv/bin/python` and `$PWD/.venv/bin/accelerate` if present, and
fall back to `python3` / `accelerate` on `PATH`. From outside the repository:

```bash
export MALIANG_ROOT=$PWD
export PYTHONPATH=$MALIANG_ROOT/code/gc25d:$MALIANG_ROOT/code/arope
export AROPE_ROOT=$MALIANG_ROOT/code/arope
export DIFFSYNTH_SKIP_DOWNLOAD=True
```

## Data

Download the game recordings used in the paper and the Wan2.2 base model, then
prepare the fxcond_v1 training data:

```
data/
├── recordings/data/<subset>/<scene>/<run>/frames-*.tar
├── gc25d_v10_fxcond/
│   ├── index/clips.jsonl
│   ├── latents/<clip>.pt
│   ├── cond/<clip>.npz
│   ├── meta/
│   └── text_table_gc25d.pt
models/
└── Wan2.2-TI2V-5B/
```

The release zip ships `assets/text_table_gc25d.pt`; copy or symlink it to
`data/gc25d_v10_fxcond/text_table_gc25d.pt`, or pass `--text_table` explicitly.

To build the cache from raw recordings:

```bash
GC_RES=832x480 python code/gc25d/prep_v10.py \
  --src data/recordings/data \
  --out data/gc25d_v10_fxcond \
  --gpus 0,1,2,3,4,5,6,7
python code/gc25d/verify_prep.py --root data/gc25d_v10_fxcond
python code/gc25d/ext_canvas.py --root data/gc25d_v10_fxcond
```

`prep_v10.py --help` lists the filters. The paper uses the `parts_fx` vocabulary: 7 static
classes, player/boss body parts, four boss-FX classes, and `255` for unknown.

## Training the bidirectional teacher

The paper's teacher was trained for 10,667 steps with global batch 16 (micro-batch 1, gradient
accumulation 2, data-parallel), `lr=1e-5`, `cond_lr=1e-4`, 500 warmup steps.

```bash
OUT=outputs/sft_fxcond_v1 GA=2 MAX_STEPS=10667 \
  bash scripts/train_sft.sh
```

The launcher selects `configs/accelerate_zero2_8gpu_ga2.yaml`, checks that the data
resolution matches `GC_RES`, and writes `train.log` plus `step-*.safetensors` under `OUT`.
Override with `DATA=...`, `GC_RES=1280x704`, `VOCAB=parts_fx`, `RESUME=...`.

## Three-stage distillation

`gcdistill` turns the bidirectional teacher into a causal four-step student. The released
code is the fxcond_v1 version used for the paper; the recipes are in
`gcdistill/config_stage{1,2,3}.json` and the launchers.

```bash
export SFT=$PWD/models/sft_fxcond_v1_step-10667.safetensors

# stage 1: teacher forcing, 8000 steps
DISTILL_NGPU=8 SFT=$SFT OUT=$PWD/outputs/distill/fxcond_v1/stage1_tf \
  bash code/gc25d/gcdistill/run_stage1.sh

# stage 2: causal consistency distillation, 8000 steps
AR=$PWD/outputs/distill/fxcond_v1/stage1_tf/dits/dit_step8000.safetensors \
DISTILL_NGPU=8 OUT=$PWD/outputs/distill/fxcond_v1/stage2_cd \
  bash code/gc25d/gcdistill/run_stage2.sh

# stage 3: DMD, 500 steps (the released deploy weight is the step-500 EMA)
STUDENT=$PWD/outputs/distill/fxcond_v1/stage2_cd/ema/ema_step8000.safetensors \
SFT=$SFT DISTILL_NGPU=8 OUT=$PWD/outputs/distill/fxcond_v1/stage3_dmd \
  bash code/gc25d/gcdistill/run_stage3.sh
```

The launchers default to the paper's data-parallel degree and the matching DeepSpeed configs. For a smaller machine, set
`DISTILL_NGPU`, `DISTILL_GPUS`, `GRAD_ACCUM`, `ACC_CFG`, and `EFF_BATCH_OVERRIDE=1`; the
published recipe keeps the effective batch at 8.

## Inference

### Teacher, on held-out clips

```bash
python code/gc25d/sample.py \
  --ckpt models/sft_fxcond_v1_step-10667.safetensors \
  --data_root data/gc25d_v10_fxcond \
  --raw_root data/recordings/data \
  --text_table assets/text_table_gc25d.pt \
  --val_n 8 --steps 50 --out_dir outputs/teacher_eval
```

Writes `<clip>.mp4` (GT | generated | map) and `<clip>.json` with PSNR/SSIM. Use `--sem_mode unknown` for the no-condition control.

### Deployed causal student, long rollout

```bash
python code/gc25d/gcdistill/eval_rollout.py \
  --ckpt models/fxcond_v1_stage3_ema_step500.safetensors \
  --data_root data/gc25d_v10_fxcond \
  --text_table assets/text_table_gc25d.pt \
  --n_clips 10 --chain_cells 241 --vae tae \
  --out "$PWD/outputs/deploy_eval"
```

Each clip produces `<clip>/gen.mp4` (generated only) and `<clip>/rollout.mp4` (GT |
generated | map), plus `summary.json` with PSNR, ZNCC, and brightness drift. Use
`--vae wan` to avoid TAEHV.

### Your own first frame and integer maps

```bash
python code/gc25d/sample_user.py \
  --ckpt models/fxcond_v1_stage3_ema_step500.safetensors \
  --first_frame first.png \
  --ids ids.npz \
  --prompts prompts.json \
  --out outputs/user_demo --name demo --steps 50
```

`ids.npz` holds a `uint8` array of shape `[81, 480, 832]` in the `parts_fx` vocabulary;
`prompts.json` holds the 21 per-cell text prompts. `--actions` accepts a compact action
script such as `"D:8,DW:4,idle:4,atk:1,D:3"`; `--export_clip <clip_id>` exports a prepared
clip's first frame, ids, prompts, and action blocks as a template.

## Evaluation

The paper evaluates frames 1–80 on the 100-clip test set at 832×480 / 16 fps. PSNR/SSIM come
from `code/gc25d/metrics.py`; LPIPS, FVD, mIoU, and entity metrics from `experiments/metrics`.

```bash
cd experiments/metrics

# PSNR / SSIM / LPIPS / FVD / mIoU / entity metrics
python eval_common.py \
  --method ours_deploy \
  --dir ../../outputs/deploy_eval \
  --kind native \
  --out ../../outputs/metrics

# extra metrics (temporal warp, boundary, ...)
python extra_eval.py --method ours_deploy --dir ../../outputs/deploy_eval --kind native

# paired bootstrap over clips / recording runs
python bootstrap_ci.py --methods ours_deploy,solar --tag main --cluster clip
```

`--seg` defaults to `assets/seg_b2_final`; `assets/i3d_torchscript.pt` is used for FVD.
The test manifest is `experiments/testset/v1/test_clips.json`; `DATA` and `RAW` default to
`data/gc25d_v10_fxcond` and `data/recordings/data`.

## Paper experiments

- **Interface ablation** — `experiments/abl/`: `run_abl.sh` trains variants ①/② from the base
  model, `run_abl_infer.sh` generates the 100 test clips, `enc/` and `vace/` contain the
  encoder and VACE follow-ups. The released ablation checkpoints are in `models/`.
- **Edit while playing** — `experiments/exp4/exp4_edit.py` switches a world-anchored edit on at
  cell 10 and scores alignment, preserved region, and latency against a same-seed base
  rollout.
- **Unseen maps** — `experiments/exp6/`: `procmap.py` renders procedural worlds from the test
  cameras, `sample_user.py` renders them, `exp6_score.py` computes mIoU/LPIPS/FVD.
- **FX controls** — `experiments/exp6/exp_fx_*.py` and `experiments/exp4/exp_fx_*.py` for natural and
  injected boss-FX layouts.

The experiment scripts assume the layout above and write under `outputs/`. They are the
reference implementations of the paper's runs, not a one-command reproduction: the raw
recordings, the SolarRenderer/AnyGameRenderer/CWM baselines, and the FLUX first-frame
generator are external.

## Runtime

The 5B teacher uses DeepSpeed ZeRO-2; stage 3 uses FSDP1. Inference runs on a single GPU with enough
memory for the 5B DiT in bf16 plus activations. The deployed student is a 4-step causal model; with TAEHV
and a 17-cell KV window it streams in real time on one GPU.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. The checkpoints are released for research use;
third-party components (Wan2.2, DiffSynth-Studio, LPIPS, SegFormer, I3D) keep their
upstream licenses.

## Citation

A BibTeX entry will be added once the paper is public.

## Downloads

- **Code + assets:** [release v1.0](https://github.com/YeyingLab/MaLiang/releases/tag/v1.0)
  (`maliang-v1-code.zip`, `maliang-v1-code+assets.zip`).
- **Checkpoints:** [release v1.0-models](https://github.com/YeyingLab/MaLiang/releases/tag/v1.0-models),
  26 parts. Reassemble and verify with:

```bash
bash scripts/download_models.sh
```

The five files are described in [Model zoo](#model-zoo) and in `models.sha256`.
