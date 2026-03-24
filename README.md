# HuggingFace Model Recommender

**Find which AI models will actually run on your machine.**

Detects your hardware (CPU, RAM, GPU/Apple Silicon), scores hundreds of HuggingFace text-generation models across **quality, speed, fit, and context** dimensions, and tells you exactly which ones will run well — and which won't.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What's Included](#whats-included)
3. [Installation](#installation)
4. [Usage — Notebook Version](#usage--notebook-version)
5. [Usage — Command-Line Version](#usage--command-line-version)
6. [Configuration Options](#configuration-options)
7. [Understanding the Output](#understanding-the-output)
8. [Scoring System Explained](#scoring-system-explained)
9. [Understanding the Results Table](#understanding-the-results-table)
10. [Understanding Recommendations](#understanding-recommendations)
11. [Model Formats & Quantization Guide](#model-formats--quantization-guide)
12. [Hardware Detection Details](#hardware-detection-details)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)

---

## Quick Start

**macOS / Linux:**
```bash
# 1. Copy this folder to your machine
# 2. Run the setup script (creates venv + installs deps)
bash setup.sh

# 3. Activate and run
source venv/bin/activate
python3 recommender.py
```

**Windows (Command Prompt or PowerShell):**
```cmd
REM 1. Copy this folder to your machine
REM 2. Run the setup script (creates venv + installs deps)
setup.bat

REM 3. Activate and run
venv\Scripts\activate
python recommender.py
```

> **Note:** `hf_models.json` is already included in this folder. If it's missing, copy it from the parent directory.

That's it. You'll see your hardware profile, a ranked table of models, and specific recommendations.

---

## What's Included

| File | Purpose |
|------|---------|
| `recommender.py` | Standalone command-line tool (no Jupyter needed) |
| `requirements.txt` | Python dependencies |
| `README.md` | This instruction manual |
| `hf_models.json` | Pre-built database of 593 models (copy from parent dir) |
| `setup.sh` | One-command setup script (macOS/Linux) |
| `setup.bat` | One-command setup script (Windows) |
| `../hugging_face_list.ipynb` | Interactive Jupyter notebook version (in parent dir) |

---

## Installation

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** (Python package manager)

### Step-by-step (manual install)

**macOS / Linux:**
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Optional: Install PyTorch for GPU/MPS detection
pip install torch
# For Apple Silicon: torch already includes MPS support
# For NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: Install huggingface_hub for live API mode
pip install huggingface_hub
```

**Windows (Command Prompt):**
```cmd
REM Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

REM Install required packages
pip install -r requirements.txt

REM Optional: Install PyTorch for GPU detection
pip install torch
REM For NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121

REM Optional: Install huggingface_hub for live API mode
pip install huggingface_hub
```

### One-command setup (alternative)

| OS | Command |
|----|--------|
| macOS/Linux | `bash setup.sh` |
| Windows | `setup.bat` |

Both scripts create a virtual environment, install dependencies, and attempt to install PyTorch.

### Verify installation

```bash
python recommender.py --help
```

---

## Usage — Notebook Version

The Jupyter notebook (`hugging_face_list.ipynb`) offers an interactive, visual experience:

### Opening the notebook

```bash
# Option A: VS Code (recommended)
code ../hugging_face_list.ipynb

# Option B: Jupyter Lab
pip install jupyterlab
jupyter lab ../hugging_face_list.ipynb

# Option C: Classic Jupyter
pip install notebook
jupyter notebook ../hugging_face_list.ipynb
```

### Running the notebook

Run cells **top-to-bottom** (Shift+Enter in each cell, or "Run All"):

| Cell | What it does |
|------|-------------|
| **Cell 1** (Markdown) | Title and overview — just read |
| **Cell 2** (Code) | Detects your CPU, RAM, GPU/MPS — prints a hardware summary |
| **Cell 3** (Code) | Loads models from JSON or API — configure `DATA_SOURCE` at top |
| **Cell 4** (Code) | Scores all models — prints verdict breakdown |
| **Cell 5** (Code) | Shows color-coded results table — configure filters at top |
| **Cell 6** (Code) | Shows top recommendations + quick-start commands |

### Configuring the notebook

**Cell 3 — Data Source** (near the top):
```python
DATA_SOURCE = "json"    # Fast, uses bundled hf_models.json (593 models)
DATA_SOURCE = "api"     # Fetches live from HuggingFace (needs internet)
```

**Cell 5 — Filters** (near the top):
```python
MIN_PARAMS_B    = 0       # Only show models >= this many billion params
MAX_PARAMS_B    = 999     # Only show models <= this many billion params
MIN_CONTEXT     = 0       # Only show models with context >= this
QUANT_FILTER    = "all"   # "all" | "full" | "quantized" | "gguf" | "gptq" | "awq"
TOP_N           = 50      # Number of rows to display
```

**Cell 3 — Pipeline Filter** (near the top):
```python
PIPELINE_FILTER = "text-generation"   # Default: only text-generation models
PIPELINE_FILTER = "all"               # Include all pipeline types
```

---

## Usage — Command-Line Version

### Basic usage

```bash
python recommender.py                           # Use bundled JSON, show top 50
```

### All options

```bash
python recommender.py --help

# Data source
python recommender.py --source json                       # Local JSON (default)
python recommender.py --source api --limit 200            # Live from HuggingFace

# Filters
python recommender.py --quant gguf                        # Only GGUF models
python recommender.py --quant full                        # Only full-precision
python recommender.py --quant quantized                   # All quantized types
python recommender.py --min-params 3 --max-params 14      # 3B to 14B only
python recommender.py --min-context 32768                 # 32K+ context only
python recommender.py --top 20                            # Show top 20 only
python recommender.py --pipeline all                      # Include all pipeline types
python recommender.py --pipeline text2text-generation     # Specific pipeline type

# Export
python recommender.py --export results.csv                # Save to CSV
python recommender.py --quant gguf --export gguf_models.csv

# Combine filters
python recommender.py --quant gguf --min-params 7 --max-params 14 --top 10
```

---

## Configuration Options

| Option | CLI Flag | Notebook Variable | Default | Description |
|--------|----------|-------------------|---------|-------------|
| Data source | `--source` | `DATA_SOURCE` | `json` | `json` or `api` |
| JSON file path | `--json-path` | `JSON_PATH` | `hf_models.json` | Path to model database |
| API fetch limit | `--limit` | `MODEL_LIMIT` | 500 | Models to fetch in API mode |
| Top N results | `--top` | `TOP_N` | 50 | Max results to display |
| Quant filter | `--quant` | `QUANT_FILTER` | `all` | Filter by quantization type |
| Min params (B) | `--min-params` | `MIN_PARAMS_B` | 0 | Minimum billions of parameters |
| Max params (B) | `--max-params` | `MAX_PARAMS_B` | 999 | Maximum billions of parameters |
| Min context | `--min-context` | `MIN_CONTEXT` | 0 | Minimum context window |
| Pipeline tag | `--pipeline` | `PIPELINE_FILTER` | `text-generation` | Filter by pipeline type (`all` for no filter) |
| Export CSV | `--export` | — | — | Save results to CSV file |

---

## Understanding the Output

The tool produces output in 4 sections. Here's what each means:

### Section 1: Hardware Profile

```
=======================================================
  SYSTEM HARDWARE PROFILE
=======================================================
  CPU        : arm
  Cores      : 12 physical / 12 logical
  RAM        : 32.0 GB total / 14.9 GB available
  Device     : MPS  (Apple Silicon (MPS))
  Model VRAM : 22.4 GB budget / 10.4 GB free
=======================================================
```

| Field | Meaning |
|-------|---------|
| **CPU** | Your processor architecture (`arm` = Apple Silicon, `x86_64` = Intel/AMD) |
| **Cores** | Physical cores (real) vs logical cores (with hyperthreading) |
| **RAM** | Total system memory / currently free memory |
| **Device** | What the model will run on: `CUDA` (NVIDIA GPU), `MPS` (Apple Silicon GPU), or `CPU` |
| **Model VRAM** | How much memory is available for loading models. **This is the key number.** |

**How VRAM budget is calculated:**
- **NVIDIA GPU**: Actual GPU VRAM (e.g., 24 GB on RTX 4090)
- **Apple Silicon (MPS)**: 70% of total RAM (unified memory is shared with OS)
- **CPU-only**: 80% of total RAM

### Section 1b: Top 10 RAM-Consuming Processes

```
  RAM in use: 18.3 GB (57.2% of total)
-------------------------------------------------------
  TOP 10 RAM-CONSUMING PROCESSES
-------------------------------------------------------
  Process                            PID   RAM (GB)  % of Used
  ------------------------------ ------- ---------- ----------
  Code Helper (Plugin)             87705      2.70      14.8%
  fileproviderd                     1624      1.01       5.5%
  Microsoft Teams WebView Helper   15921      0.86       4.7%
  Code Helper (Plugin)             63731      0.77       4.2%
  Code Helper (Renderer)           63329      0.75       4.1%
  ...
                                         ────────── ──────────
  Top 10 total                                8.43      46.1%

  💡 Close heavy processes above to free RAM and fit larger models.
```

Shows the 10 processes consuming the most RAM on your system right now.
- **RAM (GB)**: Resident Set Size — actual physical memory used by the process
- **% of Used**: What fraction of your used RAM this process accounts for

**Why this matters:** On Apple Silicon (unified memory) and CPU-only systems, every GB consumed by other apps is a GB less for model loading. Closing heavy processes (browsers, IDEs, Teams) and re-running the tool will increase your VRAM budget and may unlock larger models.

### Section 2: Model Loading Summary

```
Parsed 559 text-generation model variants from 593 repos.
  (34 non-text-generation models filtered out)
  Full           : 29
  GGUF           : 404
  GPTQ           : 18
  AWQ            : 45
  MLX            : 63
  TOTAL          : 559
```

Shows how many models were loaded and their breakdown by format type.

### Section 3: Scoring Summary

```
Scoring complete for 559 model variants.
  VRAM budget: 10.4 GB (MPS)
  Weights: Fit=0.4, Quality=0.25, Speed=0.25, Context=0.1
  Excellent: 348
  Good: 35
  Marginal: 5
  Won't Run: 171
```

| Verdict | What it means |
|---------|---------------|
| **Excellent** | Model fits comfortably, will run well. **Use these.** |
| **Good** | Model fits but with moderate headroom. Should work fine. |
| **Marginal** | Tight fit — might work but could be slow or OOM. Test carefully. |
| **Won't Run** | Model is too large for your hardware. Don't attempt unless you reduce quality. |

### Section 4: Results Table & Recommendations

The results table shows the top models ranked by composite score, with recommendations for the best model in each category.

---

## Scoring System Explained

Every model gets four scores (0–100 each), combined into a weighted composite:

### Fit Score (40% weight) — *"Will it actually run?"*

The most important score. Compares the model's estimated VRAM requirement against your available VRAM.

| VRAM Usage | Fit Score | Meaning |
|-----------|-----------|---------|
| ≤ 50% of budget | **100** | Fits with plenty of room — fast loading, room for KV cache |
| 50–70% | **90** | Comfortable — good headroom |
| 70–85% | **75** | Moderate — will work but less room for long contexts |
| 85–95% | **55** | Tight — may work, watch for OOM on long prompts |
| 95–100% | **35** | Very tight — likely to OOM under load |
| 100–120% | **15** | Probably won't load at all |
| > 120% | **0** | Definitely won't fit |

*Additional: CPU-only systems get a -20 penalty for models >13B (too slow to be practical).*

### Quality Score (25% weight) — *"Is this model any good?"*

Based on community signals from HuggingFace:

| Component | Weight | Source |
|-----------|--------|--------|
| Downloads | 60% | Log-scaled, measured against the 90th percentile of all models |
| Likes | 30% | Log-scaled, measured against the 90th percentile |
| Trending | 10% | Current trending score (API mode only) |

*A model with millions of downloads and thousands of likes (like Llama-3) will score near 100. A niche model with 500 downloads and 2 likes might score 20.*

### Speed Score (25% weight) — *"How fast will inference be?"*

Estimates relative inference speed on your hardware:

| Factor | Effect |
|--------|--------|
| **Model size** | Smaller models relative to others → higher score (inverse ratio to largest model) |
| **Quantization bonus** | GGUF/GPTQ/AWQ: +15 points (quantized models process faster) |
| **INT8 bonus** | 8-bit quantized: +8 points |
| **Device bonus** | CUDA: +5, MPS: +3 (GPU acceleration) |

*A 3B GGUF model on a CUDA GPU might score 95. A 70B FP16 model on CPU might score 5.*

### Context Score (10% weight) — *"How much text can it process at once?"*

| Context Window | Score | Practical Use |
|---------------|-------|---------------|
| ≤ 2K tokens | 20 | Very short conversations only |
| 4K | 40 | Basic chat, short documents |
| 8K | 60 | Standard use cases |
| 16K | 70 | Medium documents |
| 32K | 80 | Long documents, multiple files |
| 64K | 87 | Books, large codebases |
| 128K | 93 | Very long documents |
| > 128K | 100 | Entire repositories, book-length |

### Composite Score

```
Composite = (Fit × 0.40) + (Quality × 0.25) + (Speed × 0.25) + (Context × 0.10)
```

**Fit is weighted highest (40%)** because a model you can't run has zero practical value, no matter how good it is.

---

## Understanding the Results Table

### Column Reference

| Column | Description | Example |
|--------|-------------|---------|
| **Model** | HuggingFace model ID (author/name) | `Qwen/Qwen3-4B-Instruct-2507` |
| **Type** | Model format type | `GGUF`, `Full`, `GPTQ`, `AWQ`, `MLX` |
| **Quant** | Specific quantization level | `Q4_K_M`, `FP16`, `4bit` |
| **Params(B)** | Parameter count in billions | `7.24` = 7.24 billion params |
| **VRAM(GB)** | Estimated VRAM needed to load the model | `3.9` = needs ~3.9 GB |
| **Context** | Maximum context window (tokens) | `131,072` = 128K tokens |
| **Downloads** | HuggingFace download count | `15,432,100` |
| **Quality** | Quality score (0–100) | Higher = more trusted by community |
| **Speed** | Speed score (0–100) | Higher = faster inference on your hardware |
| **Fit** | Fit score (0–100) | **Most important.** Higher = fits better in your VRAM |
| **CtxScore** | Context score (0–100) | Higher = longer context window |
| **Score** | Composite score (0–100) | **Overall ranking.** Sorted by this. |
| **Verdict** | Human-readable recommendation | `Excellent`, `Good`, `Marginal`, `Won't Run` |

### Color Coding (Notebook version)

| Color | Score Range | Verdict |
|-------|------------|---------|
| 🟢 Dark green | 80–100 | Excellent |
| 🟢 Green | 60–79 | Good |
| 🟡 Orange | 40–59 | Marginal |
| 🔴 Red | 0–39 | Won't Run |

### How to Read a Row

Example row:
```
Model: Qwen/Qwen3-4B-Instruct-2507 | Type: GGUF | Quant: Q4_K_M
Params: 4.0B | VRAM: 2.1GB | Context: 262,144
Quality: 89 | Speed: 98 | Fit: 100 | CtxScore: 100 | Score: 96.7 | Verdict: Excellent
```

**Translation:** This is a 4-billion parameter model in GGUF Q4_K_M format. It needs only 2.1 GB of VRAM (your system has 10.4 GB free), so it fits easily (Fit=100). It's very popular (Quality=89), will run fast (Speed=98), and has a massive 262K token context (CtxScore=100). Overall score: 96.7/100 — **Excellent choice.**

---

## Understanding Recommendations

The tool picks the best model in each category:

| Category | What it means |
|----------|---------------|
| **Best Overall** | Highest composite score across all model types |
| **Best Full-Precision** | Best non-quantized model (FP16/BF16) — highest quality but needs more VRAM |
| **Best GGUF** | Best GGUF quantized model — ideal for llama.cpp, runs on CPU too |
| **Best GPTQ/AWQ** | Best GPU-optimized quantized model — great for NVIDIA cards |

### Hardware Utilization Insight

Shows approximately what model sizes fit at different quantization levels:

```
With 10.4 GB available on MPS:
  FP16 (full)  : up to ~4.5B params    → Phi-3-mini, small models only
  GGUF Q8      : up to ~9.0B params    → Llama-3-8B, Qwen3-8B
  GGUF Q4_K_M  : up to ~16.1B params   → Qwen2-14B, larger models
```

**Rule of thumb:** Going from FP16 → Q4_K_M lets you run a model roughly 3.5× larger on the same hardware.

---

## Model Formats & Quantization Guide

### What is quantization?

Quantization reduces model precision from 16-bit floats to smaller types, dramatically reducing memory usage with minimal quality loss.

### Format Comparison

| Format | Precision | VRAM per 7B model | Quality Loss | Speed | Best Backend |
|--------|-----------|-------------------|-------------|-------|-------------|
| **FP16** (Full) | 16-bit float | ~14 GB | None | Baseline | transformers |
| **BF16** (Full) | BFloat16 | ~14 GB | None | Baseline | transformers |
| **GPTQ** | 4-bit int | ~3.5 GB | Very small | Fast (GPU) | transformers + auto-gptq |
| **AWQ** | 4-bit int | ~3.5 GB | Very small | Fast (GPU) | transformers + auto-awq |
| **GGUF Q8_0** | 8-bit | ~7 GB | Negligible | Good | llama.cpp |
| **GGUF Q5_K_M** | ~5-bit | ~4.8 GB | Very small | Good | llama.cpp |
| **GGUF Q4_K_M** | ~4.5-bit | ~3.9 GB | Small | Good | llama.cpp |
| **GGUF Q3_K_M** | ~3.5-bit | ~2.9 GB | Moderate | Fast | llama.cpp |
| **GGUF Q2_K** | ~2.5-bit | ~2.2 GB | Significant | Fastest | llama.cpp |
| **MLX** | 4-bit | ~3.5 GB | Very small | Fast (MPS) | mlx-lm |

### Which format should you use?

- **NVIDIA GPU**: GPTQ or AWQ for best speed, or GGUF for flexibility
- **Apple Silicon (MPS)**: GGUF (with Metal acceleration) or MLX
- **CPU only**: GGUF (via llama.cpp) — only practical option for usable speeds
- **Maximum quality**: Full precision (FP16/BF16) if you have enough VRAM

---

## Hardware Detection Details

### How the tool detects your hardware

| Component | How it's detected | Library |
|-----------|-------------------|---------|
| CPU model | `platform.processor()` | Built-in |
| CPU cores | `os.cpu_count()`, `psutil.cpu_count()` | psutil |
| RAM | `psutil.virtual_memory()` | psutil |
| NVIDIA GPU | `torch.cuda.get_device_properties()` | PyTorch |
| Apple Silicon | `torch.backends.mps.is_available()` | PyTorch |
| GPU VRAM | `torch.cuda.mem_get_info()` | PyTorch |

### What if PyTorch isn't installed?

The tool still works! Without PyTorch:
- Device is set to `CPU`
- VRAM budget = 80% of RAM (models load into system RAM)
- GPU detection is skipped

For most accurate results, install PyTorch to detect your GPU/MPS properly.

### VRAM Budget Logic

| Device | VRAM Budget | Why |
|--------|-------------|-----|
| NVIDIA GPU | Actual free VRAM | Dedicated GPU memory |
| Apple Silicon | 70% of available RAM | Unified memory shared with OS — can't use 100% |
| CPU only | 80% of available RAM | Need headroom for OS and model overhead |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'psutil'"
```bash
pip install psutil
```

### "ModuleNotFoundError: No module named 'huggingface_hub'"
Only needed for API mode. Either:
```bash
pip install huggingface_hub
```
Or use JSON mode (default): `--source json`

### "ConnectError: Connection reset by peer" (API mode)
Your network is blocking connections to `huggingface.co`. Solutions:
- Use JSON mode instead: `--source json` (recommended for corporate networks)
- Connect to a different network (home WiFi, mobile hotspot)
- Contact IT about whitelisting `huggingface.co`

### "JSON file not found"
Copy `hf_models.json` to the same directory as `recommender.py`:
```bash
# macOS/Linux
cp /path/to/hf_models.json .

# Windows
copy \path\to\hf_models.json .
```

### "No GPU detected" on a machine with GPU
Install PyTorch with the correct backend:
```bash
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS is included by default)
pip install torch
```

### All models show "Won't Run"
Your available VRAM/RAM might be very low. Try:
- Close other applications to free memory
- Filter for smaller models: `--max-params 3`
- Filter for heavily quantized models: `--quant gguf`

---

## FAQ

**Q: How accurate are the VRAM estimates?**
Within ~10-20% of actual usage. The tool adds a 15% overhead factor for KV cache and runtime activations. Real usage varies with context length, batch size, and backend.

**Q: Can I update the model database?**
Yes. Set `DATA_SOURCE = "api"` (notebook) or `--source api` (CLI) to fetch the latest models from HuggingFace. You need internet access and the `huggingface_hub` package. You can also export API results to update your JSON: fetch via API, copy the scored data to a new JSON.

**Q: Why does a 7B model show different VRAM for different quant levels?**
Because quantization changes how many bytes each parameter uses:
- 7B × 2 bytes (FP16) = 14 GB
- 7B × 0.56 bytes (Q4_K_M) = 3.9 GB
Same model, ~3.5× less memory.

**Q: I have 16 GB RAM on Apple Silicon. What's the biggest model I can run?**
Your effective VRAM budget is ~11.2 GB (70% of 16 GB). You can run:
- Up to ~4.9B in FP16
- Up to ~9.7B in GGUF Q8
- Up to ~17.4B in GGUF Q4_K_M

**Q: Why is Fit weighted 40% — the highest?**
A model with perfect quality, speed, and context scores is useless if it won't load on your machine. Fit is the gatekeeper. The remaining 60% differentiates among models that *can* run.

**Q: Can I change the scoring weights?**
Yes. In the notebook, edit Cell 4 (Scoring Engine):
```python
W_FIT     = 0.40   # Increase to prioritize "will it fit?"
W_QUALITY = 0.25   # Increase to prioritize popular/trusted models
W_SPEED   = 0.25   # Increase to prioritize faster inference
W_CONTEXT = 0.10   # Increase to prioritize longer context windows
```
Weights must sum to 1.0.

**Q: What's the difference between the notebook and CLI version?**
Both use the same scoring logic. The notebook adds:
- Color-coded interactive table (green/yellow/orange/red)
- Rich Markdown recommendations
- Visual styling
- Can re-run individual cells to adjust filters without re-fetching

The CLI adds:
- No Jupyter dependency
- CSV export
- Command-line arguments for scripting/automation
- Works in pure terminal environments

---

## Sharing with Your Team

### Option A: Share the folder
1. Copy the `hf-model-recommender/` folder (includes everything)
2. Send via Teams/email/shared drive
3. Teammates run:

**macOS/Linux:**
```bash
bash setup.sh
source venv/bin/activate
python3 recommender.py
```

**Windows:**
```cmd
setup.bat
venv\Scripts\activate
python recommender.py
```

### Option B: Share via Git
```bash
cd hf-model-recommender
git init
git add .
git commit -m "HuggingFace Model Recommender tool"
# Push to your team's repo
```

### Option C: Share the notebook
1. Copy `hugging_face_list.ipynb` + `hf_models.json` together
2. Teammates open in VS Code or Jupyter and run all cells

---

*Built for the team. Runs on macOS, Linux, and Windows — anywhere Python does.*
