# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # wwd_i — Phase 2: train the backbone on Colab (T4 or CPU)
#
# Trains the from-scratch **BC-ResNet** backbone with **cosine-prototypical** metric
# learning, runs the **few-shot probe gate** on held-out (unseen) words, and exports
# a frozen `backbone.onnx`.
#
# The corpus is **MSWC** (Multilingual Spoken Words): hundreds/thousands of words
# streamed from the Hugging Face hub, materialized as a capped subset, then trained.
# Streaming pulls only the subset we keep, not the full multi-GB corpus.
#
# **GPU optional but recommended:** *Runtime → Change runtime type → **T4 GPU***
# makes training far faster, but a **CPU runtime also works** (just slower) — every
# cell auto-detects the device. Stays at project parity — **Python 3.14 + torch 2.12**
# via `uv` — installing the **CUDA** torch build when a GPU is present, else the CPU
# build (`--torch-backend=auto`).

# %% [markdown]
# ### 1. (optional) GPU — a CPU runtime works too
#

# %%
# !nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo 'no GPU — CPU runtime is fine (training just slower)'

# %% [markdown]
# ### 2. Get the code
#
# `git clone`s the public repo. Private fork? Add a `GITHUB_TOKEN` in **Colab
# Secrets** (🔑 left sidebar) and it's injected into the clone URL automatically —
# no token in the notebook.

# %%
import os

from google.colab import userdata

REPO_URL = "https://github.com/m0nochr0me/wwd_i.git"
BRANCH = "master"
# Private repo? Add a GITHUB_TOKEN in Colab Secrets (🔑 left sidebar) — it's injected
# here, never pasted into the notebook. Public repo: leave the secret unset.
try:
    _tok = userdata.get("GITHUB_TOKEN")
    if _tok:
        REPO_URL = REPO_URL.replace("https://", f"https://{_tok}@")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):  # ruff-format strips these parens -> SyntaxError on py3  # fmt: skip
    pass
# HF_TOKEN (Colab Secrets) -> env so MSWC streaming authenticates to the HF hub
# (higher rate limits; required if the dataset is gated). Optional — unset = anonymous.
try:
    _hf = userdata.get("HF_TOKEN")
    if _hf:
        os.environ["HF_TOKEN"] = _hf
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):  # ruff-format strips these parens -> SyntaxError on py3  # fmt: skip
    pass
os.environ["REPO_URL"] = REPO_URL
os.environ["BRANCH"] = BRANCH
os.chdir("/content")  # chdir to root before deleting
# !rm -rf /content/wwd_i && git clone --branch "$BRANCH" --depth 1 "$REPO_URL" /content/wwd_i
# %cd /content/wwd_i
# !git log --oneline -1

# %% [markdown]
# ### 3. Provision Python 3.14 with `uv`
#

# %%
# !pip install -q uv
# !uv venv --python 3.14
# !.venv/bin/python --version

# %% [markdown]
# ### 4. Install torch (CUDA if a GPU is present, else CPU) + the package
#
# `--no-config` ignores the repo's CPU torch pin; `--torch-backend=auto` detects the
# runtime — CUDA wheels for a T4 (fallback: `--index-url https://download.pytorch.org/whl/cu124`),
# or CPU wheels on a CPU runtime. `datasets` is the MSWC streaming dep.
#

# %%
# !uv pip install --no-config --python .venv/bin/python torch torchaudio --torch-backend=auto
# our package (numpy/onnx/onnxruntime/soundfile/soxr) + the ONNX exporter + MSWC streaming (script-based -> datasets<4)
# !uv pip install --python .venv/bin/python -e . onnxscript 'datasets<4'

# %% [markdown]
# ### 5. Verify torch — show the device
#

# %%
# !.venv/bin/python -c "import torch; cuda = torch.cuda.is_available(); print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0) if cuda else 'CPU (no GPU — training runs but is much slower)')"

# %% [markdown]
# ### 6. Google Drive — persist data across runtime crashes
#
# Mounts Drive and defines restore/backup helpers. The MSWC subset (a long HF stream) is archived to Drive as `mswc.tar` and restored on a fresh runtime, and the trained `backbone_mswc.onnx/.pt` are mirrored out — so a disconnect doesn't cost you the re-stream or the training run.

# %%
# --- Google Drive persistence: consistent data dir + restore/backup ---
import os
import shutil
import tarfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
DATA = "/content/data"  # consistent corpus dir (already used by the cells below)
DRIVE = "/content/drive/MyDrive/wwd_i"  # Drive mirror for the expensive-to-recompute artifacts
ART = "/content/artifacts"
for d in (DATA, DRIVE, ART):
    os.makedirs(d, exist_ok=True)
os.environ["DATA"], os.environ["DRIVE"], os.environ["ART"] = DATA, DRIVE, ART


def backup(path: str) -> None:
    """Mirror a local file -> Drive (by basename)."""
    src = Path(path)
    if not src.exists():
        print(f"skip backup: {src} missing")
        return
    shutil.copy2(src, Path(DRIVE) / src.name)
    print(f"backed up {src.name} to Drive")


def backup_tar(local_dir: str) -> None:
    """Archive a large many-file dir -> Drive as <name>.tar (fast restore vs. copytree)."""
    d = Path(local_dir)
    if not d.exists():
        print(f"skip backup: {d} missing")
        return
    tar = Path(DRIVE) / f"{d.name}.tar"
    with tarfile.open(tar, "w") as t:
        t.add(d, arcname=d.name)
    print(f"backed up {d.name} -> {tar}")


def restore_tar(name: str, dst_parent: str) -> bool:
    """Extract DRIVE/<name>.tar into dst_parent if present. Returns True if restored."""
    tar = Path(DRIVE) / f"{name}.tar"
    if not tar.exists():
        return False
    with tarfile.open(tar, "r") as t:
        t.extractall(dst_parent)
    print(f"restored {name} from {tar}")
    return True


print("DATA =", DATA, "| DRIVE =", DRIVE, "| ART =", ART)

# %% [markdown]
# ## MSWC — stream, materialize, train
#
# Streams `MLCommons/ml_spoken_words` from the HF hub, materializes a capped subset
# of 16 kHz clips, then trains. Streaming means we pull only the subset we keep, not
# the full multi-GB corpus.

# %% [markdown]
# ### A1. Inspect the dataset schema (do this first)
#
# Prints one streamed example's fields and decodes one clip. The default config is
# `en_opus` (English; much smaller to stream than `en_wav`). Confirm `decoded OK`
# appears and the word field is `keyword`. If opus fails to decode, add `--config
# en_wav` to A1/A2 (larger but plain wav); if the word field differs, pass
# `--word-key`.

# %%
# !.venv/bin/python -m wwd_i.data.mswc --inspect

# %% [markdown]
# ### A2. Materialize a subset
#
# Words are **hash-selected** (uniform across the alphabet), each filled to
# `--clips-per-word`, so the vocabulary spans `a`–`z`, not just `a`. This streams the
# **whole** `en_opus` split once (the cost of alphabet coverage — allow a good while
# and some bandwidth), writing only the ~`--n-words` selected words. Faster options:
# `--split validation` (smaller split, full alphabet, fewer clips/word) or
# `--max-stream 200000` (partial alphabet). Bump `--vocab-total` if the kept count
# comes out far below `--n-words`. Written under `$DATA/mswc` and **restored from
# Drive** (as `mswc.tar`) if a previous run already materialized it — skipping the
# long stream after a crash.

# %%
# Materialize the MSWC subset, or restore it from Drive (skips the long HF stream after a crash).
if not restore_tar("mswc", DATA):
    # !.venv/bin/python -m wwd_i.data.mswc --root $DATA/mswc --n-words 500 --clips-per-word 80
    backup_tar(f"{DATA}/mswc")

# %% [markdown]
# ### A2b. Noise pool for additive-noise augmentation (recommended)
#
# `--augment` does reverb + speed with no extra data, but the highest-value piece —
# **additive background noise** — needs a pool. This random-samples `--n-clips` clips
# (seeded, reproducible) from a large ambient corpus
# (`benjamin-paine/freesound-laion-640k`: cars, bells, birds, …) by **streaming**, so it
# pulls only the subset it keeps, not the full 640k. The sampled wavs are archived to
# Drive (`noise_pool.tar`) and restored on a fresh runtime.
#
# Swap `--dataset` for any HF audio set (e.g. a MUSAN mirror); add `--inspect` first if
# unsure of the audio column. Licence note: Freesound clips are mixed CC (some
# non-commercial) — fine as training-only noise, mind it if you redistribute artifacts.

# %%
# Noise pool for --aug-bg-dir: random-sample N clips (seeded) from a large ambient
# corpus, or restore the sampled subset from Drive. Streaming + seeded shuffle pulls
# only ~--n-clips, not the whole 640k. Unsure of the audio column? Inspect first:
# #   !.venv/bin/python -m wwd_i.data.hf_audio --inspect --dataset benjamin-paine/freesound-laion-640k
if not restore_tar("noise_pool", DATA):
    # !.venv/bin/python -m wwd_i.data.hf_audio --dataset benjamin-paine/freesound-laion-640k --n-clips 2000 --seed 0 --out $DATA/noise_pool
    backup_tar(f"{DATA}/noise_pool")

# %% [markdown]
# ### A3. Train the backbone (with augmentation)
#
# Bigger vocabulary → harder episodes (`--n-way 30`) and more steps. `--augment`
# perturbs **training** clips on the fly (reverb / speed-perturb / additive noise /
# clipping) so the embedding is robust to real capture conditions; the held-out
# probe stays **clean**, so its number stays comparable to the un-augmented run
# (frozen baseline ≈ **0.87**). Watch `[probe @ N]`: it should still clearly beat
# the mel-baseline — a small dip vs 0.87 is fine if robustness is the goal.
#
# Additive noise (the highest-value augmentation) needs a pool: add
# `--aug-bg-dir <dir>` pointing at a noise corpus (e.g. **MUSAN**). Without it only
# reverb+speed+clip apply — gain is skipped on purpose (the backbone z-score
# normalizes it away). Device is auto-selected (CUDA else CPU). Augmentation now runs **batched on
# the GPU** (it was CPU-side and throttled throughput), and decoded clips are cached
# in RAM, so the otherwise-idle GPU does the work and stays fed.

# %%
# !.venv/bin/python -m wwd_i.train.train_backbone --dataset mswc --root $DATA/mswc --augment --aug-bg-dir $DATA/noise_pool --mswc-held-out 50 --n-way 30 --probe-n-way 10 --k-shot 5 --q-query 5 --steps 8000 --eval-every 500 --probe-episodes 200 --embedding-dim 96 --out $ART/backbone_mswc.pt --onnx $ART/backbone_mswc.onnx
backup(f"{ART}/backbone_mswc.onnx")
backup(f"{ART}/backbone_mswc.pt")

# %% [markdown]
# ### A4. Download the MSWC artifacts
#
# (Already mirrored to Drive by the training cell; these `files.download` calls just pull them to your laptop.)

# %%
from google.colab import files

files.download(f"{ART}/backbone_mswc.onnx")
files.download(f"{ART}/backbone_mswc.pt")

# %% [markdown]
# ## Interpreting the gate
#
# - **Pass:** backbone probe accuracy clearly above the mel-baseline on held-out
#   words → the embedding generalizes → drop `backbone_mswc.onnx` into the repo as
#   the frozen backbone and move to Phase 3 (per-word data) / Phase 4 (head).
# - **Weak / not clearly above the mel-baseline:** widen vocabulary (`--n-words`),
#   add clips (`--clips-per-word`), raise `--embedding-dim`, or train longer.
