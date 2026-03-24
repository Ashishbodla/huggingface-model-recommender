#!/usr/bin/env python3
"""
HuggingFace Model Recommender — Standalone CLI Version
======================================================
Detects your hardware, scores HuggingFace text-generation models across
quality, speed, fit, and context dimensions, and tells you which ones
will actually run well on your machine.

Usage:
    python recommender.py                        # Use bundled JSON (default)
    python recommender.py --source api           # Fetch live from HuggingFace
    python recommender.py --source json --json-path my_models.json
    python recommender.py --top 30 --quant gguf  # Show top 30 GGUF models
    python recommender.py --min-params 3 --max-params 14  # 3B-14B only
    python recommender.py --export results.csv   # Export to CSV
"""

import argparse
import json
import math
import os
import platform
import re
import sys

import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BYTES_PER_PARAM = {
    "FP32": 4.0, "FP16": 2.0, "BF16": 2.0,
    "INT8": 1.0, "8bit": 1.0,
    "INT4": 0.5, "4bit": 0.5,
    "GPTQ": 0.5, "AWQ": 0.5,
    "GPTQ-Int4": 0.5, "GPTQ-Int8": 1.0,
    "AWQ-4bit": 0.5, "AWQ-8bit": 1.0,
    "Q2_K": 0.31, "Q3_K_S": 0.38, "Q3_K_M": 0.41, "Q3_K_L": 0.44,
    "Q4_0": 0.50, "Q4_K_S": 0.53, "Q4_K_M": 0.56,
    "Q5_0": 0.63, "Q5_K_S": 0.66, "Q5_K_M": 0.69,
    "Q6_K": 0.81, "Q8_0": 1.0, "IQ2_XS": 0.28, "IQ3_XS": 0.36,
}
OVERHEAD_FACTOR = 1.15

# ---------------------------------------------------------------------------
# 1. Hardware Detection
# ---------------------------------------------------------------------------
def detect_hardware():
    cpu_model = platform.processor() or platform.machine()
    cpu_cores_logical = os.cpu_count()
    cpu_cores_physical = psutil.cpu_count(logical=False)

    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 1)
    ram_available_gb = round(mem.available / (1024**3), 1)

    device_type = "cpu"
    gpu_name = None
    vram_total_gb = 0.0
    vram_available_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            device_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_total_gb = round(props.total_mem / (1024**3), 1)
            vram_free, _ = torch.cuda.mem_get_info(0)
            vram_available_gb = round(vram_free / (1024**3), 1)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            gpu_name = "Apple Silicon (MPS)"
            vram_total_gb = round(ram_total_gb * 0.70, 1)
            vram_available_gb = round(ram_available_gb * 0.70, 1)
        else:
            vram_total_gb = round(ram_total_gb * 0.80, 1)
            vram_available_gb = round(ram_available_gb * 0.80, 1)
    except ImportError:
        vram_total_gb = round(ram_total_gb * 0.80, 1)
        vram_available_gb = round(ram_available_gb * 0.80, 1)

    hw = {
        "cpu_model": cpu_model,
        "cpu_cores_logical": cpu_cores_logical,
        "cpu_cores_physical": cpu_cores_physical,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "device_type": device_type,
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_available_gb": vram_available_gb,
    }

    print("=" * 55)
    print("  SYSTEM HARDWARE PROFILE")
    print("=" * 55)
    print(f"  CPU        : {cpu_model}")
    print(f"  Cores      : {cpu_cores_physical} physical / {cpu_cores_logical} logical")
    print(f"  RAM        : {ram_total_gb} GB total / {ram_available_gb} GB available")
    device_line = f"  Device     : {device_type.upper()}"
    if gpu_name:
        device_line += f"  ({gpu_name})"
    print(device_line)
    print(f"  Model VRAM : {vram_total_gb} GB budget / {vram_available_gb} GB free")
    print("=" * 55)
    if device_type == "cpu":
        print("  ** No GPU detected - models will run on CPU (slower).")
        print("     Quantized GGUF models via llama.cpp recommended.")
    elif device_type == "mps":
        print("  Apple Silicon - unified memory shared with OS.")
        print("  Budget set to ~70% of RAM for model loading.")
    print("=" * 55)

    # --- Top 10 RAM-consuming processes ---
    ram_used_gb = round(ram_total_gb - ram_available_gb, 1)
    print(f"\n  RAM in use: {ram_used_gb} GB ({round(ram_used_gb/ram_total_gb*100, 1)}% of total)")
    print("-" * 55)
    print("  TOP 10 RAM-CONSUMING PROCESSES")
    print("-" * 55)
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            mi = p.info.get('memory_info')
            if mi is None:
                continue
            mem_gb = mi.rss / (1024**3)
            if mem_gb > 0.01:  # Skip trivial processes
                procs.append((p.info['name'] or '?', p.info['pid'], mem_gb))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
            pass
    procs.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'Process':<30s} {'PID':>7s} {'RAM (GB)':>10s} {'% of Used':>10s}")
    print(f"  {'-'*30} {'-'*7} {'-'*10} {'-'*10}")
    for name, pid, mem_gb in procs[:10]:
        pct = round(mem_gb / ram_used_gb * 100, 1) if ram_used_gb > 0 else 0
        print(f"  {name:<30s} {pid:>7d} {mem_gb:>9.2f}  {pct:>8.1f}%")
    top_total = sum(m for _, _, m in procs[:10])
    print(f"  {'':30s} {'':>7s} {'─'*10} {'─'*10}")
    print(f"  {'Top 10 total':<30s} {'':>7s} {top_total:>9.2f}  {round(top_total/ram_used_gb*100,1) if ram_used_gb > 0 else 0:>8.1f}%")
    print(f"\n  💡 Close heavy processes above to free RAM and fit larger models.")
    print("=" * 55)
    print()
    return hw

# ---------------------------------------------------------------------------
# 2. Model Loading — JSON
# ---------------------------------------------------------------------------
def load_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    rows, skipped = [], 0
    for m in data:
        fmt = (m.get("format") or "").lower()
        quant_raw = m.get("quantization", "")

        if fmt == "gguf":
            qtype, qlevel = "GGUF", quant_raw or "Q4_K_M"
        elif fmt == "gptq":
            qtype, qlevel = "GPTQ", quant_raw or "4bit"
        elif fmt == "awq":
            qtype, qlevel = "AWQ", quant_raw or "4bit"
        elif fmt == "mlx":
            qtype, qlevel = "MLX", quant_raw or "4bit"
        else:
            qtype, qlevel = "Full", "FP16"

        param_count = m.get("parameters_raw", 0)
        params_b = round(param_count / 1e9, 2) if param_count else None

        est_vram = m.get("min_vram_gb")
        if not est_vram and param_count:
            bpp = BYTES_PER_PARAM.get(qlevel, BYTES_PER_PARAM.get(qtype, 2.0))
            est_vram = round((param_count * bpp / (1024**3)) * OVERHEAD_FACTOR, 2)

        name = m.get("name", "")
        row = {
            "model_id": name,
            "author": m.get("provider", name.split("/")[0] if "/" in name else ""),
            "downloads": m.get("hf_downloads", 0),
            "likes": m.get("hf_likes", 0),
            "trending": 0,
            "params_b": params_b,
            "quant_type": qtype,
            "quant_level": qlevel,
            "est_vram_gb": est_vram,
            "context_len": m.get("context_length"),
            "license": None,
            "library": None,
            "gguf_file": None,
            "pipeline_tag": m.get("pipeline_tag", "text-generation"),
        }
        if row["est_vram_gb"] is not None or row["params_b"] is not None:
            rows.append(row)
        else:
            skipped += 1
    return rows, len(data), skipped

# ---------------------------------------------------------------------------
# 2b. Model Loading — HuggingFace API
# ---------------------------------------------------------------------------
def load_from_api(limit, task_filter):
    from huggingface_hub import HfApi
    api = HfApi()

    print(f"Fetching top {limit} '{task_filter}' models from HuggingFace Hub...")
    print("(This may take 15-30 seconds with expanded metadata)\n")

    raw_models = list(api.list_models(
        filter=task_filter, sort="downloads", limit=limit,
        expand=["safetensors", "gguf", "config", "tags",
                "cardData", "trendingScore", "siblings"],
    ))

    def _detect_quant(model):
        tags = [t.lower() for t in (model.tags or [])]
        name_lower = model.modelId.lower()
        siblings = model.siblings or []
        filenames = [s.rfilename for s in siblings]
        param_count = None
        if model.safetensors and hasattr(model.safetensors, "parameter_count"):
            pc = model.safetensors.parameter_count
            param_count = sum(pc.values()) if isinstance(pc, dict) else (int(pc) if isinstance(pc, (int, float)) else None)
        results = []
        gguf_files = [f for f in filenames if f.lower().endswith(".gguf")]
        if gguf_files or "gguf" in tags:
            for fname in gguf_files:
                match = re.search(r"((?:IQ|Q)\d[A-Za-z0-9_]*)", fname, re.IGNORECASE)
                qlevel = match.group(1).upper() if match else "Q4_K_M"
                bpp = BYTES_PER_PARAM.get(qlevel, 0.56)
                file_size_gb = None
                for s in siblings:
                    if s.rfilename == fname and hasattr(s, "size") and s.size:
                        file_size_gb = s.size / (1024**3)
                        break
                est = round(file_size_gb * OVERHEAD_FACTOR, 2) if file_size_gb else (round((param_count * bpp / (1024**3)) * OVERHEAD_FACTOR, 2) if param_count else None)
                results.append(("GGUF", qlevel, est, fname))
            if not results and param_count:
                results.append(("GGUF", "Q4_K_M", round((param_count * 0.56 / (1024**3)) * OVERHEAD_FACTOR, 2), None))
            return results, param_count
        if "gptq" in tags or "gptq" in name_lower:
            est = round((param_count * 0.5 / (1024**3)) * OVERHEAD_FACTOR, 2) if param_count else None
            return [("GPTQ", "4bit", est, None)], param_count
        if "awq" in tags or "awq" in name_lower:
            est = round((param_count * 0.5 / (1024**3)) * OVERHEAD_FACTOR, 2) if param_count else None
            return [("AWQ", "4bit", est, None)], param_count
        dtype = "FP16"
        if model.safetensors and hasattr(model.safetensors, "parameter_count"):
            pc = model.safetensors.parameter_count
            if isinstance(pc, dict) and pc:
                dominant = max(pc, key=pc.get).upper()
                dtype = {"F32": "FP32", "F16": "FP16", "BF16": "BF16", "I8": "INT8", "I4": "INT4"}.get(dominant, dominant)
        bpp = BYTES_PER_PARAM.get(dtype, 2.0)
        est = round((param_count * bpp / (1024**3)) * OVERHEAD_FACTOR, 2) if param_count else None
        return [("Full", dtype, est, None)], param_count

    def _get_ctx(model):
        cfg = model.config or {}
        if isinstance(cfg, dict):
            inner = cfg
            if len(cfg) == 1 and isinstance(list(cfg.values())[0], dict):
                inner = list(cfg.values())[0]
            for key in ["max_position_embeddings", "max_sequence_length", "n_positions",
                        "seq_length", "sliding_window", "max_seq_len", "model_max_length"]:
                val = inner.get(key) or cfg.get(key)
                if val and isinstance(val, (int, float)):
                    return int(val)
        return None

    def _get_license(model):
        for tag in (model.tags or []):
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
        return None

    rows, skipped = [], 0
    for m in raw_models:
        quant_variants, param_count = _detect_quant(m)
        ctx_len = _get_ctx(m)
        lic = _get_license(m)
        params_b = round(param_count / 1e9, 2) if param_count else None
        for qtype, qlevel, est_vram, gguf_file in quant_variants:
            row = {
                "model_id": m.modelId, "author": m.author,
                "downloads": m.downloads or 0, "likes": m.likes or 0,
                "trending": getattr(m, "trending_score", None) or 0,
                "params_b": params_b, "quant_type": qtype, "quant_level": qlevel,
                "est_vram_gb": est_vram, "context_len": ctx_len,
                "license": lic, "library": m.library_name, "gguf_file": gguf_file,
                "pipeline_tag": "text-generation",
            }
            if row["est_vram_gb"] is not None or row["params_b"] is not None:
                rows.append(row)
            else:
                skipped += 1
    return rows, len(raw_models), skipped

# ---------------------------------------------------------------------------
# 3. Scoring Engine
# ---------------------------------------------------------------------------
def score_models(model_rows, hardware):
    W_FIT, W_QUALITY, W_SPEED, W_CONTEXT = 0.40, 0.25, 0.25, 0.10

    all_downloads = [r["downloads"] for r in model_rows if r["downloads"] > 0]
    all_likes = [r["likes"] for r in model_rows if r["likes"] > 0]
    all_params = [r["params_b"] for r in model_rows if r["params_b"]]

    log_dl = np.log1p(all_downloads) if all_downloads else np.array([0])
    log_lk = np.log1p(all_likes) if all_likes else np.array([0])
    dl_p90 = float(np.percentile(log_dl, 90)) if len(log_dl) > 1 else 1.0
    lk_p90 = float(np.percentile(log_lk, 90)) if len(log_lk) > 1 else 1.0
    max_params = max(all_params) if all_params else 70.0

    vram_budget = hardware["vram_available_gb"]
    device = hardware["device_type"]

    def _quality(r):
        dl = min(math.log1p(r["downloads"]) / dl_p90, 1.0) * 60
        lk = min(math.log1p(r["likes"]) / lk_p90, 1.0) * 30
        tr = min(r["trending"] / 50, 1.0) * 10 if r["trending"] else 0
        return round(dl + lk + tr, 1)

    def _speed(r):
        p = r["params_b"]
        if not p:
            return 30.0
        base = max(0, (1 - p / max_params) * 80)
        qb = 15 if r["quant_type"] in ("GGUF", "GPTQ", "AWQ") else (8 if r["quant_level"] in ("INT8", "8bit") else 0)
        db = 5 if device == "cuda" else (3 if device == "mps" else 0)
        return round(min(base + qb + db, 100), 1)

    def _fit(r):
        v = r["est_vram_gb"]
        if v is None:
            return 20.0
        if vram_budget <= 0:
            return 5.0
        ratio = v / vram_budget
        if ratio <= 0.50:    score = 100.0
        elif ratio <= 0.70:  score = 90.0
        elif ratio <= 0.85:  score = 75.0
        elif ratio <= 0.95:  score = 55.0
        elif ratio <= 1.0:   score = 35.0
        elif ratio <= 1.2:   score = 15.0
        else:                score = 0.0
        if device == "cpu" and r.get("params_b") and r["params_b"] > 13:
            score = max(score - 20, 0)
        return score

    def _context(r):
        c = r["context_len"]
        if not c: return 30.0
        if c <= 2048: return 20.0
        if c <= 4096: return 40.0
        if c <= 8192: return 60.0
        if c <= 16384: return 70.0
        if c <= 32768: return 80.0
        if c <= 65536: return 87.0
        if c <= 131072: return 93.0
        return 100.0

    def _verdict(fit):
        if fit >= 85: return "Excellent"
        if fit >= 60: return "Good"
        if fit >= 30: return "Marginal"
        return "Won't Run"

    for r in model_rows:
        r["s_quality"] = _quality(r)
        r["s_speed"] = _speed(r)
        r["s_fit"] = _fit(r)
        r["s_context"] = _context(r)
        r["composite"] = round(
            W_FIT * r["s_fit"] + W_QUALITY * r["s_quality"] +
            W_SPEED * r["s_speed"] + W_CONTEXT * r["s_context"], 1)
        r["verdict"] = _verdict(r["s_fit"])

    model_rows.sort(key=lambda r: r["composite"], reverse=True)

    verdicts = {}
    for r in model_rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    print(f"Scoring complete for {len(model_rows)} model variants.")
    print(f"  VRAM budget: {vram_budget} GB ({device.upper()})")
    print(f"  Weights: Fit={W_FIT}, Quality={W_QUALITY}, Speed={W_SPEED}, Context={W_CONTEXT}")
    for v in ["Excellent", "Good", "Marginal", "Won't Run"]:
        print(f"  {v}: {verdicts.get(v, 0)}")
    print()
    return model_rows

# ---------------------------------------------------------------------------
# 4. Display
# ---------------------------------------------------------------------------
def display_results(model_rows, hardware, top_n=50, quant_filter="all",
                    min_params=0, max_params=999, min_context=0, export_path=None):
    filtered = model_rows.copy()
    if min_params > 0:
        filtered = [r for r in filtered if r["params_b"] and r["params_b"] >= min_params]
    if max_params < 999:
        filtered = [r for r in filtered if r["params_b"] and r["params_b"] <= max_params]
    if min_context > 0:
        filtered = [r for r in filtered if r["context_len"] and r["context_len"] >= min_context]

    qf = quant_filter.lower()
    if qf == "full":
        filtered = [r for r in filtered if r["quant_type"] == "Full"]
    elif qf == "quantized":
        filtered = [r for r in filtered if r["quant_type"] != "Full"]
    elif qf in ("gguf", "gptq", "awq", "mlx"):
        filtered = [r for r in filtered if r["quant_type"].lower() == qf]

    filtered = filtered[:top_n]
    df = pd.DataFrame(filtered)

    cols = ["model_id", "quant_type", "quant_level", "params_b", "est_vram_gb",
            "context_len", "downloads", "s_quality", "s_speed", "s_fit",
            "s_context", "composite", "verdict"]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()
    df_out.columns = ["Model", "Type", "Quant", "Params(B)", "VRAM(GB)",
                      "Context", "Downloads", "Quality", "Speed", "Fit",
                      "CtxScore", "Score", "Verdict"][:len(cols)]

    print(f"Showing {len(df_out)} models (filtered from {len(model_rows)} total)")
    print(f"Filters: params={min_params}-{max_params}B, context>={min_context}, quant={quant_filter}\n")
    print(df_out.to_string(index=False))
    print()

    if export_path:
        df_out.to_csv(export_path, index=False)
        print(f"Results exported to: {export_path}\n")

    # --- Recommendations ---
    def _best(label, fn):
        cands = [r for r in model_rows if fn(r) and r["verdict"] != "Won't Run"]
        if not cands:
            print(f"  {label}: No suitable models found.")
            return
        b = cands[0]
        vstr = f"{b['est_vram_gb']:.1f}GB" if b["est_vram_gb"] else "?"
        cstr = f"{b['context_len']:,}" if b["context_len"] else "?"
        backend = "llama.cpp" if b["quant_type"] == "GGUF" else "transformers"
        if b["quant_type"] in ("GPTQ", "AWQ"):
            backend = f"transformers+auto-{b['quant_type'].lower()}"
        print(f"  {label}: {b['model_id']}")
        print(f"    {b['quant_type']} {b['quant_level']} | {b['params_b']}B | VRAM:{vstr} | Ctx:{cstr}")
        print(f"    Score: {b['composite']}/100 (Fit:{b['s_fit']:.0f} Qual:{b['s_quality']:.0f} Spd:{b['s_speed']:.0f} Ctx:{b['s_context']:.0f})")
        print(f"    Backend: {backend}")

    print("=" * 60)
    print("  TOP RECOMMENDATIONS")
    print("=" * 60)
    _best("Best Overall", lambda r: True)
    print()
    _best("Best Full-Precision", lambda r: r["quant_type"] == "Full")
    print()
    _best("Best GGUF", lambda r: r["quant_type"] == "GGUF")
    print()
    _best("Best GPTQ/AWQ", lambda r: r["quant_type"] in ("GPTQ", "AWQ"))
    print()

    # Hardware insight
    vram = hardware["vram_available_gb"]
    fp16_max = round(vram / (2.0 * OVERHEAD_FACTOR), 1)
    q4_max = round(vram / (0.56 * OVERHEAD_FACTOR), 1)
    q8_max = round(vram / (1.0 * OVERHEAD_FACTOR), 1)

    print("=" * 60)
    print("  HARDWARE UTILIZATION INSIGHT")
    print("=" * 60)
    print(f"  With {vram:.1f} GB available on {hardware['device_type'].upper()}:")
    print(f"    FP16 (full)  : up to ~{fp16_max}B params")
    print(f"    GGUF Q8      : up to ~{q8_max}B params")
    print(f"    GGUF Q4_K_M  : up to ~{q4_max}B params")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Recommender — find models that fit your hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recommender.py                              # Use bundled JSON
  python recommender.py --source api --limit 200     # Fetch live from HF
  python recommender.py --quant gguf --top 20        # Top 20 GGUF models
  python recommender.py --min-params 7 --max-params 14
  python recommender.py --export results.csv         # Save to CSV
        """)
    parser.add_argument("--source", choices=["json", "api"], default="json",
                        help="Data source: 'json' (local file) or 'api' (HuggingFace Hub)")
    parser.add_argument("--json-path", default="hf_models.json",
                        help="Path to local JSON model file (default: hf_models.json)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of models to fetch in API mode (default: 500)")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top results to display (default: 50)")
    parser.add_argument("--quant", default="all",
                        help="Filter: all|full|quantized|gguf|gptq|awq|mlx")
    parser.add_argument("--min-params", type=float, default=0,
                        help="Minimum parameter count in billions")
    parser.add_argument("--max-params", type=float, default=999,
                        help="Maximum parameter count in billions")
    parser.add_argument("--min-context", type=int, default=0,
                        help="Minimum context window length")
    parser.add_argument("--pipeline", default="text-generation",
                        help="Filter by pipeline_tag (default: text-generation). Use 'all' to include all types.")
    parser.add_argument("--export", default=None,
                        help="Export results to CSV file path")
    args = parser.parse_args()

    # Step 1: Hardware
    hardware = detect_hardware()

    # Step 2: Load models
    if args.source == "json":
        json_path = args.json_path
        if not os.path.exists(json_path):
            # Try looking in the script's directory
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.json_path)
        if not os.path.exists(json_path):
            print(f"ERROR: JSON file not found: {args.json_path}")
            print("  Place hf_models.json in the same directory as this script,")
            print("  or use --json-path to specify the path, or use --source api.")
            sys.exit(1)
        print(f"Loading models from: {json_path}")
        model_rows, total, skipped = load_from_json(json_path)
        print(f"Source: Local JSON ({total} models in file)")
    else:
        model_rows, total, skipped = load_from_api(args.limit, "text-generation")
        print(f"Source: HuggingFace API (fetched {total} repos)")

    # Filter by pipeline_tag
    pre = len(model_rows)
    if args.pipeline.lower() != "all":
        model_rows = [r for r in model_rows if r.get("pipeline_tag", "text-generation") == args.pipeline]
    filt = pre - len(model_rows)

    pipeline_label = args.pipeline if args.pipeline.lower() != "all" else "all pipelines"
    print(f"\nParsed {len(model_rows)} {pipeline_label} models from {total} repos.")
    if filt:
        print(f"  ({filt} non-{args.pipeline} models filtered out)")
    if skipped:
        print(f"  ({skipped} skipped - insufficient metadata)")
    for qt in ["Full", "GGUF", "GPTQ", "AWQ", "MLX"]:
        c = sum(1 for r in model_rows if r["quant_type"] == qt)
        if c:
            print(f"  {qt:15s}: {c}")
    print(f"  {'TOTAL':15s}: {len(model_rows)}\n")

    # Step 3: Score
    model_rows = score_models(model_rows, hardware)

    # Step 4: Display
    display_results(model_rows, hardware, top_n=args.top, quant_filter=args.quant,
                    min_params=args.min_params, max_params=args.max_params,
                    min_context=args.min_context, export_path=args.export)


if __name__ == "__main__":
    main()
