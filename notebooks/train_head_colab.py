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
#     language: python
#     name: python3
# ---

# %% [markdown]
# # wwd_i — Phases 3-4: train a wake-word head on Colab
#
# Generates per-word data with **ElevenLabs v3**, embeds it through the **frozen `backbone.onnx`** already in the repo, trains a tiny **streaming GRU head**, calibrates the threshold to **FA < 0.5/hr & FR < 5%**, and exports `<word>_head.onnx`.
#
# Change `WAKE_WORD` and re-run to train a different word — the backbone is reused; only the head is retrained. Each word's data and head are backed up under its own slug on Drive, so different wake words don't collide.
#
# A **GPU is optional but recommended**: the heavy step is embedding tens of thousands of background crops through the backbone, which runs on the GPU via `onnxruntime-gpu` (installed automatically in step 4 when a GPU is present) — without one it falls back to CPU. Add your **`ELEVENLABS_API_KEY`** to **Colab Secrets** (🔑 left sidebar); it's read in step 6.
#
# **Runtime crash-safe:** data and the preprocessed background cache are mirrored to Google Drive and auto-restored (step 5), so a disconnect doesn't cost you the ElevenLabs spend or a multi-GB re-download.

# %% [markdown]
# ### 1. (optional) GPU

# %%
# !nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'no GPU — fine, the head is tiny'

# %% [markdown]
# ### 2. Get the code
#
# `git clone`s the public repo. Private fork? Use `https://<TOKEN>@github.com/m0nochr0me/wwd_i.git`.

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
# ### 3. Python 3.14 via uv

# %%
# !pip install -q uv
# !uv venv --python 3.14
# !.venv/bin/python --version

# %% [markdown]
# ### 4. Install torch + the package + the ElevenLabs SDK
#
# `--torch-backend=auto` picks a CUDA torch wheel if a GPU is present, else CPU. On a
# GPU runtime the cell then swaps the CPU `onnxruntime` for **`onnxruntime-gpu`** so the
# frozen backbone embeds on the GPU — that's the heavy part (embedding tens of thousands
# of background crops in Part B½). No GPU: it stays on CPU (the head itself is tiny).
#
# `onnxruntime-gpu`'s CUDA major is **pinned to match torch's** (read from `torch.version.cuda`):
# CUDA 12 → `onnxruntime-gpu<1.27`, CUDA 13 → latest. **1.27 dropped CUDA 12** and links
# against `libcudart.so.13`, so an *unpinned* install on a CUDA-12 torch runtime fails to
# import (`ImportError: libcudart.so.13`). With the majors matched, onnxruntime reuses the
# CUDA/cuDNN libs torch already installed. The training scripts request
# `CUDAExecutionProvider` with a CPU fallback and print the active providers, so a silent
# fallback to CPU is visible.

# %%
# !uv pip install --no-config --python .venv/bin/python torch torchaudio --torch-backend=auto
# !uv pip install --python .venv/bin/python -e . onnxscript elevenlabs google-genai

# GPU runtime? Swap the CPU onnxruntime wheel for onnxruntime-gpu so the frozen
# backbone embeds on the GPU — the heavy preprocess_bg (Part B½) and the
# positives/hard-negs (Part C) then run on CUDA instead of CPU. No GPU: stay on CPU.
#
# onnxruntime-gpu's CUDA major MUST match the torch wheel --torch-backend=auto installed:
# onnxruntime-gpu 1.27+ dropped CUDA 12 and links against CUDA 13 (libcudart.so.13), so an
# unpinned install on a CUDA-12 torch runtime imports-errors with
#   ImportError: libcudart.so.13: cannot open shared object file
# Read torch.version.cuda and pin onnxruntime-gpu to the same line: <1.27 is the last
# CUDA-12 build, latest (>=1.27) is CUDA 13. onnxruntime then reuses the CUDA/cuDNN libs
# torch already installed (importing torch preloads them).
import shutil
import subprocess

# shutil.which() guards the FileNotFoundError that subprocess.run(["nvidia-smi"]) raises on a
# CPU-only runtime, where the binary isn't installed at all (not merely a non-zero exit code).
if shutil.which("nvidia-smi") and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
    cuda_major = subprocess.run(
        [".venv/bin/python", "-c", "import torch; print((torch.version.cuda or '').split('.')[0])"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = "onnxruntime-gpu" if cuda_major == "13" else "onnxruntime-gpu<1.27"  # <1.27 = CUDA 12
    print(f"GPU detected (torch CUDA {cuda_major or '?'}) -> installing {spec}")
    get_ipython().system("uv pip uninstall --python .venv/bin/python onnxruntime")
    get_ipython().system(f"uv pip install --no-config --python .venv/bin/python '{spec}'")
    # import torch first so onnxruntime reuses torch's already-loaded CUDA/cuDNN libs; a
    # CPU-only providers list here means the libs still didn't load (the diagnostic to chase).
    get_ipython().system(
        ".venv/bin/python -c \"import torch, onnxruntime as ort; print('ORT providers:', ort.get_available_providers())\""
    )
else:
    print("no GPU -> staying on CPU onnxruntime (the head is tiny)")

# %% [markdown]
# ### 5. Google Drive — persist data across runtime crashes
#
# Mounts Drive and fixes a **consistent data dir at `/content/data`** (outside the repo, so re-running the clone cell never wipes it). `restore(name)` pulls an artifact back from Drive if a previous run saved it; `backup(name)` mirrors it out.
#
# Word-specific artifacts (positives, hard-negatives, the trained head) are namespaced under the wake word's slug — `MyDrive/wwd_i/<slug>/…` — so training a **different** wake word never overwrites or wrongly restores another's. The word-**independent** background cache (`bg_neg.npy`, `noise_pool`) is saved once with `shared=True` at the top level and reused for every word. The expensive things — ElevenLabs clips (cost money) and the preprocessed cache — are saved once and restored instantly if the runtime dies.

# %%
# --- Google Drive persistence: consistent data dir + per-word restore/backup ---
import os
import shutil
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
DATA = "/content/data"  # consistent, OUTSIDE the repo -> survives re-running the clone cell
DRIVE = "/content/drive/MyDrive/wwd_i"  # Drive mirror for the expensive-to-recompute artifacts
os.makedirs(DATA, exist_ok=True)
os.makedirs(DRIVE, exist_ok=True)
os.environ["DATA"], os.environ["DRIVE"] = DATA, DRIVE


def _drive_path(name: str, shared: bool) -> Path:
    """Where `name` lives in Drive. Word-specific artifacts (positives, hard-negs,
    the head) are namespaced under the wake word's slug so a different word never
    overwrites or wrongly restores another's; shared=True keeps the word-INDEPENDENT
    background cache at the top level, reused across every word."""
    base = Path(DRIVE) if shared else Path(DRIVE) / os.environ["WORD_SLUG"]
    return base / name


def restore(name: str, shared: bool = False) -> bool:
    """Copy DRIVE[/slug]/name -> DATA/name if present in Drive. Returns True if restored."""
    src, dst = _drive_path(name, shared), Path(DATA) / name
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"restored {name} from Drive")
    return True


def backup(name: str, shared: bool = False) -> None:
    """Mirror DATA/name -> DRIVE[/slug]/name for crash recovery."""
    src, dst = Path(DATA) / name, _drive_path(name, shared)
    if not src.exists():
        print(f"skip backup: {src} missing")
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"backed up {name} to Drive")


print("DATA =", DATA, "| DRIVE =", DRIVE)

# %% [markdown]
# ### 6. Configure — wake word + API key
#
# Set the wake word here. The **`ELEVENLABS_API_KEY`** is read from **Colab Secrets** (🔑 in the left sidebar) via `userdata.get` — add it there once and grant this notebook access; it's never pasted into the notebook or written to disk. The slug derived from the wake word keys this run's per-word Drive backups (step 5).

# %%
import os

from google.colab import userdata

WAKE_WORD = "hey computer"  # <-- the phrase to detect
os.environ["WAKE_WORD"] = WAKE_WORD
os.environ["WORD_SLUG"] = WAKE_WORD.replace(" ", "_")
# Key from Colab Secrets (🔑 left sidebar) — never pasted into the notebook or written to disk.
os.environ["ELEVENLABS_API_KEY"] = userdata.get("ELEVENLABS_API_KEY")
assert os.environ["ELEVENLABS_API_KEY"], "add ELEVENLABS_API_KEY in Colab Secrets (🔑) and grant this notebook access"
# GEMINI_API_KEY (Colab Secrets) is OPTIONAL — only A3's --llm-confusables uses it; unset ->
# A3 falls back to the generic confusables. Same optional-secret pattern as GITHUB_TOKEN/HF_TOKEN.
os.environ["GEMINI_API_KEY"] = ""
try:
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY") or ""
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):  # ruff-format strips these parens -> SyntaxError on py3  # fmt: skip
    pass
print("wake word:", WAKE_WORD, "| LLM confusables:", "on" if os.environ["GEMINI_API_KEY"] else "off")

# %% [markdown]
# ## Part A — positives & hard negatives (ElevenLabs v3)

# %% [markdown]
# ### A1. Smoke-test the SDK (one clip)
#
# Validates your key and the SDK surface **before** spending on a batch. If it errors on the API call, the fix is usually a small tweak to `_synthesize`/`_client` in `wwd_i/data/elevenlabs.py` to match your installed SDK version.

# %%
# !.venv/bin/python -m wwd_i.data.elevenlabs --phrase "$WAKE_WORD" --out $DATA/smoke --smoke

# %% [markdown]
# ### A2. Generate positives (~300 diverse clips)
#
# Many voices × prosody settings; cached, so re-running is cheap and bumping `--n-clips` only adds new variants.

# %%
# positives — restore from Drive if a prior run saved them, else generate (paid) and back up
if not restore("pos"):
    # !.venv/bin/python -m wwd_i.data.elevenlabs --phrase "$WAKE_WORD" --out $DATA/pos --n-clips 300
    backup("pos")

# %% [markdown]
# ### A3. Generate hard negatives (near phrases)
#
# The wake phrase's sub-words + generic confusables (`negatives.hard_negative_phrases`), **plus** — when `GEMINI_API_KEY` is set (step 6) — N **acoustic confusables** of the wake phrase from Gemini (`--llm-confusables N`, `data.confusables`): near-homophones / rhymes that actually sound like the wake word, the hardest false-trigger source. No key → just the generic confusables.

# %%
# hard negatives — same restore-or-generate-then-backup pattern.
# --llm-confusables N adds N Gemini near-homophones of the wake phrase (the hardest negatives);
# auto-enabled only when GEMINI_API_KEY is set (step 6), else generic confusables only.
CONFUSABLES = "--llm-confusables 20" if os.environ.get("GEMINI_API_KEY") else ""
if not restore("hardneg"):
    # !.venv/bin/python -m wwd_i.data.elevenlabs --hard-negs-for "$WAKE_WORD" --out $DATA/hardneg --n-clips 20 $CONFUSABLES
    backup("hardneg")

# %% [markdown]
# ## Part B — background negatives: noise, music, speech & vocal bursts
#
# AudioSet (noise/general) + FMA (music) + **MSWC (single-word speech)** + **vocal bursts
# (cough/laugh/breath)** are decoded **once** into a compact negative-embedding cache in
# the next section, so training never holds the raw corpus in RAM. The speech negatives
# (B3) are what let the threshold drop below ~0.995 without FR exploding; the vocal bursts
# (B4) target non-speech false accepts. These downloads are **skipped automatically if the
# cache (`bg_neg.npy`) already exists** locally or on Drive — the raw audio is disposable
# after preprocessing.

# %% [markdown]
# ### B1. AudioSet — one balanced-train shard

# %%
# Download AudioSet only if the preprocessed cache doesn't already exist (local or Drive).
if os.path.exists(f"{DATA}/bg_neg.npy") or os.path.exists(f"{DRIVE}/bg_neg.npy"):
    print("bg_neg cache present — skipping AudioSet download")
else:
    # !mkdir -p $DATA/bg/audioset
    # !wget -q -O /tmp/audioset.tar 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data/bal_train09.tar?download=true'
    # !tar -xf /tmp/audioset.tar -C $DATA/bg/audioset && rm /tmp/audioset.tar
    # !echo "audioset files:"; find $DATA/bg/audioset -type f | wc -l

# %% [markdown]
# ### B2. (optional) FMA music — `fma_small` is ~7.5 GB; skip if bandwidth is tight

# %%
# Optional music corpus (~7.5 GB). Skipped if the cache exists; comment the body out to skip entirely.
if os.path.exists(f"{DATA}/bg_neg.npy") or os.path.exists(f"{DRIVE}/bg_neg.npy"):
    print("bg_neg cache present — skipping FMA download")
else:
    # !mkdir -p $DATA/bg/fma
    # !wget -q -O /tmp/fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
    # !unzip -q /tmp/fma_small.zip -d $DATA/bg/fma && rm /tmp/fma_small.zip
    # !echo "fma mp3s:"; find $DATA/bg/fma -name '*.mp3' | wc -l

# %% [markdown]
# ### B3. MSWC speech negatives (the lever for the FA/FR gate)
#
# Real-world false accepts are dominated by **speech**, but AudioSet/FMA are noise/music — so a head trained only on them can suppress FAs only by pushing the threshold toward ~0.995, where **FR explodes** (the gate-FAIL symptom). This folds **single-word human speech** from MSWC (`MLCommons/ml_spoken_words`, streamed) into the **same** `bg_neg.npy` cache, giving the head hard speech negatives so the decision boundary sharpens and the operating threshold drops.
#
# Word-independent (frozen backbone), so it's shared across every wake word; the raw clips are disposable once Part B½ embeds them (only `bg_neg.npy` is kept and backed up). Skipped automatically if the cache already exists.

# %%
# MSWC speech negatives — single non-wake words folded into the SAME bg cache.
# Real-world false accepts are dominated by SPEECH, but AudioSet/FMA are noise/music; trained on
# those alone the head can only suppress FAs by pushing the threshold to ~0.995, where FR explodes
# (the gate-FAIL symptom). MSWC adds hard speech negatives so the boundary sharpens at a lower thr.
# Skipped if the cache exists (the raw wavs are disposable once Part B½ embeds them). Materialized
# under $DATA/bg/mswc so the existing `preprocess_bg --background $DATA/bg` picks it up automatically.
if os.path.exists(f"{DATA}/bg_neg.npy") or os.path.exists(f"{DRIVE}/bg_neg.npy"):
    print("bg_neg cache present — skipping MSWC speech negatives")
else:
    # MSWC streams via a HF loading script -> needs datasets<4 (same pin as the backbone notebook).
    # !uv pip install --python .venv/bin/python 'datasets<4'
    # --n-words 20000 >= --vocab-total (20000) disables mswc.py's hash word-filter (keep =
    # hash % max(n_words,vocab_total) < n_words becomes always-true), so EVERY streamed word is kept
    # up to --clips-per-word -> ~tens of thousands of clips: max speech volume from the bounded stream
    # (negatives want volume, not vocab). --max-stream truncates the ALPHABETICAL stream mid-'a' so the
    # kept vocab is a partial alphabet -> fine for negatives (only the onset is skewed; mid/end-word
    # phones still vary). Do NOT revert to the hash default (n_words << vocab_total): that is the
    # BACKBONE's a–z spread and needs the WHOLE stream — under --max-stream it only thins the 'a'-prefix
    # to ~1k clips (the 955-clip/38-word trap), it does NOT spread the alphabet. More speech: raise
    # --max-stream (reaches further into the alphabet, slower) or --clips-per-word.
    # Single words ≠ the full wake phrase, so they are legitimate negatives even when a sub-word of the
    # phrase happens to be selected.
    # !.venv/bin/python -m wwd_i.data.mswc --root $DATA/bg/mswc --n-words 20000 --clips-per-word 30 --max-stream 300000
    # !echo "mswc speech-negative clips:"; find $DATA/bg/mswc -name '*.wav' | wc -l

# %% [markdown]
# ### B4. Vocal bursts (cough / laugh / breath) — non-speech false-accept triggers
#
# Coughs, laughs, throat-clears and breaths are classic wake-word **false accepts** that
# AudioSet/FMA/MSWC under-represent. This random-samples `--n-clips` (seeded) from
# `0x3/vocal-bursts` into `$DATA/bg/vocal_bursts`, so Part B½ folds them into the same
# `bg_neg.npy` cache as legitimate negatives. Streaming pulls only the subset kept;
# skipped automatically if the cache already exists.

# %%
# Vocal-burst negatives (cough/laugh/breath) folded into the SAME bg cache as legitimate
# negatives. Skipped if the cache exists (raw wavs disposable once Part B½ embeds them).
# Materialized under $DATA/bg/vocal_bursts so the existing `preprocess_bg --background
# $DATA/bg` picks it up automatically. Bursts are short (~0.3-1 s): keep >=0.4 s here, and
# Part B½ lowers preprocess_bg --min-seconds to 0.4 to match (else they'd be skipped).
# 0x3/vocal-bursts is a WebDataset: the FLAC bytes live in a `flac` column (its metadata
# mislabels it `audio`), so pass --audio-key flac. Unsure of a set's column? --inspect lists them.
if os.path.exists(f"{DATA}/bg_neg.npy") or os.path.exists(f"{DRIVE}/bg_neg.npy"):
    print("bg_neg cache present — skipping vocal-bursts download")
else:
    # !uv pip install --python .venv/bin/python 'datasets<4'
    # !.venv/bin/python -m wwd_i.data.hf_audio --dataset 0x3/vocal-bursts --n-clips 2000 --seed 0 --out $DATA/bg/vocal_bursts --min-seconds 0.4 --audio-key flac
    # !echo "vocal-burst clips:"; find $DATA/bg/vocal_bursts -name '*.wav' | wc -l

# %% [markdown]
# ### B5. Held-out negatives for streaming calibration
#
# Part C calibrates the threshold by streaming continuous negative audio through the **real runtime engine** (sliding the window every 80 ms + refractory debounce) — the honest FA/hr, not one max-over-time score per isolated clip. That set must be **held out** from the training negatives, so this pulls a **separate AudioSet shard** (`bal_train08`, disjoint from the training shard `bal_train09` in B1) into `$DATA/calib_bg`; it is never folded into `bg_neg.npy`. AudioSet's ~10 s clips of real audio (speech/music/noise) are exactly what the runtime slides over — more held-out hours = finer FA/hr resolution.

# %%
import os

# Held-out continuous negatives for STREAMING calibration (Part C --calib-bg). A SEPARATE
# AudioSet shard (bal_train08), disjoint from the training shard (bal_train09, B1), so it is
# never embedded into bg_neg.npy and the FA/hr gate stays honest. Re-downloaded on a fresh
# runtime rather than mirrored to Drive — it's one quick shard, not the ElevenLabs/FMA spend.
CALIB_BG = f"{DATA}/calib_bg"
if not (os.path.isdir(CALIB_BG) and os.listdir(CALIB_BG)):
    # !mkdir -p $DATA/calib_bg
    # !wget -q -O /tmp/calib.tar 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data/bal_train08.tar?download=true'
    # !tar -xf /tmp/calib.tar -C $DATA/calib_bg && rm /tmp/calib.tar
# !echo "calib negatives:"; find $DATA/calib_bg -type f | wc -l

# %% [markdown]
# ## Part B½ — preprocess backgrounds into a negative-embedding cache (once)
#
# Decodes the raw AudioSet + FMA + MSWC pool under `$DATA/bg` **file-by-file**, crops it, and embeds it through the frozen backbone into a compact `bg_neg.npy` (`[N, W, D]`, ~4 KB/clip) plus a small `noise_pool/` of raw wavs for positive augmentation. This is the fix for the old OOM: training no longer decodes the whole corpus or samples `--n-bg-neg` crops into RAM — it just loads this cache, so the negative count scales freely.
#
# The cache is **word-independent** (frozen backbone), so it's reused for every wake word, and it's backed up to Drive — restored instantly if the runtime dies. Bump `--n-bg-neg` for more negative hours (sharper FA/hr); the long music/noise files yield up to 8 crops each, while each MSWC clip is one short word, so raise it if you add a lot of speech and want it all kept.

# %%
# Build the negative-embedding cache once (or restore it from Drive). Raw $DATA/bg is disposable after this.
# shared=True -> the cache is word-independent (frozen backbone), kept at the Drive top level and reused for every word.
# --min-seconds 0.4 keeps the short vocal bursts (B4); preprocess pads them to the clip length when embedding.
r1, r2 = restore("bg_neg.npy", shared=True), restore("noise_pool", shared=True)
if not (r1 and r2):
    # !.venv/bin/python -m wwd_i.train.preprocess_bg --background $DATA/bg --out $DATA/bg_neg.npy --n-bg-neg 50000 --noise-pool-dir $DATA/noise_pool --min-seconds 0.4
    backup("bg_neg.npy", shared=True)
    backup("noise_pool", shared=True)

# %% [markdown]
# ## Part C — train the head
#
# Embeds the positives/hard-negatives through the frozen backbone, loads the precomputed background negatives from `--bg-neg-emb`, trains the GRU, and calibrates the threshold. Watch the `[gate]` line: **PASS** = FA < 0.5/hr and FR < 5%.
#
# `--background` here points at the small `noise_pool` (additive-noise augmentation for the positives); the bulk negatives come from the cache, so there's no `--n-bg-neg`/`--max-bg` decode at train time. The trained head is mirrored to Drive.
#
# `--calib-bg $DATA/calib_bg` (B5) makes calibration **honest for streaming**: instead of one max-over-time score per isolated 1.5 s negative clip, it exports the head and runs the **real engine** over the held-out continuous negatives — sliding the window every 80 ms with the refractory debounce, exactly as the runtime does — so the reported FA/hr matches deployment (the DET table is then labelled `streaming`). Streaming a few hours through the CPU engine adds a few minutes. Drop `--calib-bg` to fall back to the faster per-clip estimate.

# %%
# !mkdir -p $DATA/artifacts
# !.venv/bin/python -m wwd_i.train.train_head --word "$WAKE_WORD" --positives $DATA/pos --hard-neg $DATA/hardneg --background $DATA/noise_pool --bg-neg-emb $DATA/bg_neg.npy --calib-bg $DATA/calib_bg --n-aug 5 --epochs 40 --hidden 48 --out $DATA/artifacts/${WORD_SLUG}_head.onnx --threshold-out $DATA/artifacts/${WORD_SLUG}_head.json
backup("artifacts")

# %% [markdown]
# ## Part D — download the head + calibration

# %%
from google.colab import files

slug = os.environ["WORD_SLUG"]
files.download(f"{DATA}/artifacts/{slug}_head.onnx")
files.download(f"{DATA}/artifacts/{slug}_head.json")

# %% [markdown]
# ## Interpreting the gate
#
# - **PASS** (FA < 0.5/hr, FR < 5%): drop `<word>_head.onnx` + its `.json` (threshold + refractory) into the repo — the frozen backbone plus this head are a complete detector → Phase 5 (streaming runtime).
# - **High FA**: more / harder negatives — most FAs are speech, so scale the **MSWC** speech set (B3) first; then rebuild the cache with a bigger preprocess `--n-bg-neg` (and more background files), or add more hard-neg phrases.
# - **High FR**: more positives (`--n-clips`) or augmentation variety (`--n-aug`); confirm the A1/A2 clips are on-phrase.
# - With `--calib-bg` the FA/hr is the **streaming** rate over the **held-out** negative hours (B5) — add more held-out shards to resolve < 0.5/hr confidently. Without it, the per-clip estimate's resolution scales with the preprocess `--n-bg-neg` instead.
