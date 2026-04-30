"""HTML report rendering for REFRACT.

Single self-contained HTML page with inline CSS — no CDN deps, no JS
framework. Pastes into Discord / saves to disk / opens in any browser.

Renders the same data as :func:`refract.report.json_report` plus extra
metadata that helps reproducibility:

  - Hardware (chip, RAM, OS) detected at run time
  - Model params (file size, architecture, head count, vocab) read from
    GGUF metadata or HuggingFace ``config.json``
  - Repro command — exactly the ``argv`` that produced this report

Layout is opinionated for layman skimmability:
  1. Big composite + band
  2. Diagnosis (colored callout)
  3. Per-axis bars
  4. R-NIAH heatmap (when present)
  5. PLAD per-perturbation bars (when present)
  6. Run details (model + hardware + version stamps)
  7. Repro command (copy-paste block)
  8. Raw JSON in a collapsed `<details>` at the bottom
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json as _json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .axes.gtm import GTMResult
from .axes.kld import KLDResult
from .axes.plad import PLADResult
from .axes.rniah import RNIAHResult
from .axes.trajectory import TrajectoryResult
from .score import CompositeScore, MIN_FLOOR, band, interpret_pattern


_BAND_COLORS = {
    "EXCELLENT": "#2d8c2d",
    "PASS":      "#3aa05a",
    "DEGRADED":  "#d49327",
    "FAIL":      "#cc3232",
}


_BAND_PROSE = {
    "EXCELLENT": "Indistinguishable from the reference. Safe to deploy.",
    "PASS":      "Minor drift; safe to deploy in most uses.",
    "DEGRADED":  "Visible drift. Audit on your workload before deploying.",
    "FAIL":      "Material quality loss. Treat as broken.",
}


_AXIS_PROSE = {
    "gtm":        "Token-level agreement with the fp16 reference (greedy decode).",
    "trajectory": "Token-level agreement with the fp16 reference (greedy decode).",
    "kld":        "Distribution-level divergence from the fp16 reference (corpus KLD).",
    "rniah":      "Long-context retrieval quality vs the reference (NIAH at multiple lengths).",
    "plad":       "Robustness to small prompt changes vs the reference (typo/case/punct/paraphrase).",
}


_AXIS_LABEL = {
    "gtm":        "Axis A — GTM (Greedy Trajectory Match)",
    "trajectory": "Axis A — Trajectory (Greedy Trajectory Match, decode-time)",
    "kld":        "Axis B — KLD@D (KL Divergence at the Decoder)",
    "rniah":      "Axis C — R-NIAH (Retrieval Needle-In-A-Haystack)",
    "plad":       "Axis D — PLAD (Perturbation-Locality Aware Drift)",
}


# --------------------------------------------------------------------------
# Hardware + model metadata helpers
# --------------------------------------------------------------------------


def _hardware_metadata() -> dict:
    info: dict = {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    # macOS: chip + RAM via sysctl
    if info["system"] == "Darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if chip:
                info["chip"] = chip
        except Exception:
            pass
        try:
            memsize = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if memsize:
                # Use binary GiB (Apple's marketing convention) —
                # 137438953472 bytes / 1024**3 = 128 GiB (exact),
                # not 137.4 GB which is the misleading SI conversion.
                info["ram_gb"] = round(int(memsize) / 1024**3, 1)
        except Exception:
            pass
    # Linux: chip from /proc/cpuinfo, RAM via /proc/meminfo
    elif info["system"] == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1024 / 1024, 1)
                        break
        except Exception:
            pass
    # NVIDIA GPU info if nvidia-smi is on PATH
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if out:
            info["nvidia_gpus"] = [
                {"name": parts[0].strip(),
                 "memory_mb": int(parts[1].strip())}
                for line in out.splitlines()
                for parts in [line.split(",")] if len(parts) >= 2
            ]
    except Exception:
        pass
    return info


def _model_metadata(model_path: Path) -> dict:
    info: dict = {"path": str(model_path), "name": model_path.name}
    if not model_path.exists():
        return info
    if model_path.is_file():
        # Binary GiB — matches model-card conventions ("28 GB Q8_0").
        info["size_gb"] = round(model_path.stat().st_size / 1024**3, 2)
        info["format"] = "gguf" if model_path.suffix == ".gguf" else "file"
    elif model_path.is_dir():
        total = 0
        for ext in ("*.safetensors", "*.bin", "*.npz"):
            for f in model_path.glob(ext):
                total += f.stat().st_size
        info["size_gb"] = round(total / 1024**3, 2)
        info["format"] = "directory"
        # Read HF/MLX config.json if present
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                cfg = _json.loads(config_path.read_text())
                for k in (
                    "model_type", "architectures", "hidden_size",
                    "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "max_position_embeddings",
                    "vocab_size", "head_dim",
                ):
                    if k in cfg:
                        info[k] = cfg[k]
            except Exception:
                pass
    return info


def _repro_command(raw_json: dict | None, model: str,
                   reference_label: str, candidate_label: str,
                   has_rniah: bool, has_plad: bool) -> str:
    """Return the shell-escaped command that produced this report, with
    home-dir paths sanitized to ``~/...``.

    Priority:
      1. ``raw_json["repro_command"]`` — actual sanitized argv captured at
         score time (v0.3.2+).
      2. Fallback: walk ``sys.argv`` and sanitize home dir.
      3. Fallback (only when sys.argv looks unrelated, e.g. when
         re-rendering an old JSON via a helper script): synthesize a
         clean ``refract score ...`` command from the report fields.
    """
    # 1. Use the JSON's repro command if present (v0.3.2+).
    if raw_json and raw_json.get("repro_command"):
        return raw_json["repro_command"]

    # 2. Sanitize sys.argv on the fly.
    home = os.path.expanduser("~")
    def _sanitize(arg: str) -> str:
        if home and arg.startswith(home + os.sep):
            return "~" + arg[len(home):]
        return arg
    argv_san = [shlex.quote(_sanitize(a)) for a in sys.argv]
    # If the invocation was a refract CLI command, return it sanitized.
    if any("refract" in a for a in sys.argv):
        return " ".join(argv_san)

    # 3. Synthesize a clean stand-in (used when re-rendering old JSON via
    # a helper script that has no relation to the original CLI invocation).
    model_short = Path(model).name
    cmd = [
        "python3", "-m", "refract.cli", "score",
        "--model", shlex.quote(model_short),
        "--reference", shlex.quote(reference_label),
        "--candidate", shlex.quote(candidate_label),
        "--prompts", "refract/prompts/v0.1.jsonl",
        "--corpus", "<path/to/wiki.test.raw>",
    ]
    if has_rniah or has_plad:
        cmd.extend(["--full"])
    if has_rniah:
        cmd.extend([
            "--rniah-haystack", "<path/to/wiki.train.raw>",
            "--rniah-ctx-max", "16384",
        ])
    cmd.extend([
        "--json-out", "report.json",
        "--html-out", "report.html",
    ])
    return " ".join(cmd)


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------


def _esc(s) -> str:
    return _html.escape("" if s is None else str(s))


def _band_color(b: str) -> str:
    return _BAND_COLORS.get(b, "#666666")


def _bar(score: float, b: str) -> str:
    """A horizontal progress bar, rendered as a span."""
    pct = max(0.0, min(100.0, float(score)))
    color = _band_color(b)
    return (
        f'<div class="bar" role="progressbar" aria-valuenow="{pct:.1f}">'
        f'<div class="fill" style="width:{pct:.1f}%; background:{color};"></div>'
        f"</div>"
    )


def _axis_block(name: str, score: float, extra_html: str = "") -> str:
    b = band(score)
    color = _band_color(b)
    return (
        f'<div class="axis">'
        f'  <div class="axis-head">'
        f'    <span class="axis-label">{_esc(_AXIS_LABEL.get(name, name))}</span>'
        f'    <span class="axis-score">{score:.2f}</span>'
        f'    <span class="axis-band" style="color:{color};">{_esc(b)}</span>'
        f"  </div>"
        f"  {_bar(score, b)}"
        f'  <div class="axis-prose">{_esc(_AXIS_PROSE.get(name, ""))}</div>'
        f"  {extra_html}"
        f"</div>"
    )


def _rniah_heatmap(rniah: RNIAHResult) -> str:
    if not rniah.cells:
        return ""
    # Group by length and position so we can render a grid.
    lengths = sorted({c.length for c in rniah.cells})
    positions = sorted({c.position for c in rniah.cells})
    rows = []
    for length in lengths:
        cells_html = []
        for pos in positions:
            cell = next(
                (c for c in rniah.cells
                 if c.length == length and c.position == pos),
                None,
            )
            if cell is None:
                cells_html.append('<td class="cell empty">—</td>')
                continue
            base = cell.base_acc
            cand = cell.cand_acc
            deg = cell.degradation
            # Color: green = base==cand both succeed; red = candidate worse;
            # gray = base also fails (no info).
            if base == 0:
                bg = "#eeeeee"
                title = f"base also fails — uninformative"
            elif deg > 0:
                bg = _band_color("FAIL")
                title = f"candidate degraded by {deg:.2f}"
            else:
                bg = _band_color("EXCELLENT")
                title = "candidate matches base"
            cells_html.append(
                f'<td class="cell" style="background:{bg};" title="{_esc(title)}">'
                f"{base:.2f}/{cand:.2f}"
                f"</td>"
            )
        rows.append(
            f"<tr><th>{length}</th>" + "".join(cells_html) + "</tr>"
        )
    pos_headers = "".join(
        f'<th>{p:.2f}</th>' for p in positions
    )
    return (
        f'<div class="subblock">'
        f'<div class="subhead">R-NIAH per-cell (base / cand) — green=match, red=degraded, gray=base also fails</div>'
        f'<table class="rniah-grid">'
        f"<thead><tr><th>length\\pos</th>{pos_headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def _plad_perturb_bars(plad: PLADResult) -> str:
    rows = []
    for pert, score in plad.per_perturbation_score.items():
        if not isinstance(score, (int, float)) or math.isnan(score):
            rows.append(
                f"<tr>"
                f"<td>{_esc(pert)}</td>"
                f"<td><span class=\"skipped\">skipped</span></td>"
                f"<td>—</td>"
                f"<td>perturbation didn't apply on these prompts</td>"
                f"</tr>"
            )
            continue
        b = band(score)
        color = _band_color(b)
        rows.append(
            f"<tr>"
            f"<td>{_esc(pert)}</td>"
            f'<td>{score:.2f}</td>'
            f'<td><span class="axis-band" style="color:{color};">{b}</span></td>'
            f'<td>{_bar(score, b)}</td>'
            f"</tr>"
        )
    return (
        f'<div class="subblock">'
        f'<div class="subhead">PLAD per-perturbation</div>'
        f'<table class="plad-grid">'
        f"<thead><tr><th>perturbation</th><th>score</th><th>band</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def _details_block(model_meta: dict, hardware_meta: dict,
                   reference_label: str, candidate_label: str,
                   environment_meta: dict) -> str:
    def kv_row(k, v):
        return f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"

    model_rows = []
    for k in ("name", "size_gb", "format", "model_type", "architectures",
              "hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "max_position_embeddings",
              "vocab_size"):
        if k in model_meta:
            label = k.replace("_", " ")
            val = model_meta[k]
            if k == "size_gb":
                val = f"{val} GB"
            model_rows.append(kv_row(label, val))

    hw_rows = []
    for k in ("chip", "platform", "ram_gb", "machine", "python"):
        if k in hardware_meta:
            label = k.replace("_", " ")
            val = hardware_meta[k]
            if k == "ram_gb":
                val = f"{val} GB"
            hw_rows.append(kv_row(label, val))
    if hardware_meta.get("nvidia_gpus"):
        gpus = ", ".join(
            f"{g['name']} ({g['memory_mb']/1024:.1f} GB)"
            for g in hardware_meta["nvidia_gpus"]
        )
        hw_rows.append(kv_row("nvidia_gpus", gpus))

    env_rows = []
    for k in ("backend", "llama_cpp_commit", "llama_cpp_bin_dir",
              "mlx_lm_version", "mlx_version"):
        if k in environment_meta:
            env_rows.append(kv_row(k, environment_meta[k]))
    env_rows.insert(0, kv_row("reference", reference_label))
    env_rows.insert(1, kv_row("candidate", candidate_label))

    return (
        f'<div class="details-grid">'
        f'  <div class="details-card">'
        f'    <h3>Model</h3>'
        f'    <table>{"".join(model_rows)}</table>'
        f"  </div>"
        f'  <div class="details-card">'
        f'    <h3>Hardware</h3>'
        f'    <table>{"".join(hw_rows)}</table>'
        f"  </div>"
        f'  <div class="details-card">'
        f'    <h3>Environment</h3>'
        f'    <table>{"".join(env_rows)}</table>'
        f"  </div>"
        f"</div>"
    )


_CSS = r"""
* { box-sizing: border-box; }
body {
    font: 14px/1.5 system-ui, -apple-system, "SF Pro Text", sans-serif;
    background: #f7f7f8;
    color: #222;
    margin: 0;
    padding: 24px;
}
.report {
    max-width: 920px;
    margin: 0 auto;
    background: white;
    padding: 32px;
    border-radius: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,.06);
}
header { border-bottom: 1px solid #eee; padding-bottom: 16px; margin-bottom: 20px; }
header h1 { margin: 0 0 4px 0; font-size: 22px; }
header .subtitle { color: #666; font-size: 12px; }
section { margin: 20px 0; }
section h2 { margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase;
             letter-spacing: 0.05em; color: #666; }
.composite-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: #fafafa;
    border-radius: 8px;
    border-left: 6px solid var(--band-color, #888);
}
.composite-number { font-size: 48px; font-weight: 700; line-height: 1; }
.composite-band { font-size: 16px; font-weight: 600; letter-spacing: 0.05em; }
.composite-summary { color: #555; }
.bar { width: 100%; height: 12px; background: #eaeaea; border-radius: 6px;
       overflow: hidden; }
.bar .fill { height: 100%; }
.diagnosis {
    background: #fff8e6;
    border-left: 4px solid #d49327;
    padding: 12px 16px;
    border-radius: 4px;
}
.diagnosis ul { margin: 0; padding-left: 20px; }
.diagnosis li { margin: 4px 0; }
.axis {
    margin: 12px 0;
    padding: 12px;
    background: #fafafa;
    border-radius: 6px;
}
.axis-head {
    display: flex; gap: 16px; align-items: baseline; margin-bottom: 6px;
}
.axis-label { flex: 1; font-weight: 600; }
.axis-score { font-size: 18px; font-weight: 700; }
.axis-band { font-size: 12px; font-weight: 600; letter-spacing: 0.05em; min-width: 70px; }
.axis-prose { font-size: 12px; color: #666; margin-top: 6px; }
.subblock { margin: 16px 0 0; }
.subhead { font-size: 12px; color: #666; margin-bottom: 6px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
table th, table td { padding: 6px 8px; text-align: left; }
table.rniah-grid th, table.rniah-grid td { border: 1px solid #e0e0e0; text-align: center; }
table.rniah-grid .cell { color: white; font-weight: 600; }
table.rniah-grid .cell.empty { background: #eee; color: #999; }
table.plad-grid th { color: #666; font-weight: 500; font-size: 12px; }
.skipped { color: #999; font-style: italic; }
.details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}
.details-card {
    background: #fafafa;
    border-radius: 6px;
    padding: 12px 16px;
}
.details-card h3 {
    margin: 0 0 8px 0; font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.05em; color: #666;
}
.details-card table th { color: #888; font-weight: 500; font-size: 12px; min-width: 80px; }
.details-card table td { font-family: ui-monospace, "SF Mono", monospace; font-size: 12px; }
.repro {
    background: #1e1e1e;
    color: #f0f0f0;
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font: 12px ui-monospace, "SF Mono", monospace;
    white-space: pre-wrap;
    word-break: break-all;
}
details { margin-top: 16px; }
details summary { cursor: pointer; color: #555; font-weight: 600; padding: 6px 0; }
details pre {
    background: #1e1e1e;
    color: #f0f0f0;
    padding: 12px;
    border-radius: 6px;
    overflow: auto;
    font: 11px ui-monospace, monospace;
    max-height: 400px;
}
@media (max-width: 600px) {
    body { padding: 12px; }
    .report { padding: 20px; }
    .composite-card { flex-direction: column; align-items: flex-start; }
    .composite-number { font-size: 36px; }
}
"""


def html_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    rniah: Optional[RNIAHResult] = None,
    plad: Optional[PLADResult] = None,
    raw_json: Optional[dict] = None,
) -> str:
    """Render the report as a self-contained HTML page (string)."""
    from . import __version__
    try:
        from .runner import get_active_backend
        bk = get_active_backend()
        env_meta = bk.model_metadata(model=Path(model)) if (bk and model) else {}
    except Exception:
        env_meta = {}
    model_meta = _model_metadata(Path(model))
    hw_meta = _hardware_metadata()
    repro = _repro_command(
        raw_json=raw_json, model=model,
        reference_label=reference_label, candidate_label=candidate_label,
        has_rniah=(rniah is not None and composite.rniah_score is not None),
        has_plad=(plad is not None and composite.plad_score is not None),
    )

    band_color = _band_color(composite.band)
    diag = interpret_pattern(
        gtm_score=composite.gtm_score,
        kld_score=composite.kld_score,
        rniah_score=composite.rniah_score,
        plad_score=composite.plad_score,
    )

    axis_a_key = "trajectory" if isinstance(gtm, TrajectoryResult) else "gtm"
    axes_html = [
        _axis_block(axis_a_key, composite.gtm_score),
        _axis_block("kld", composite.kld_score),
    ]
    if rniah is not None and composite.rniah_score is not None:
        axes_html.append(_axis_block("rniah", composite.rniah_score, _rniah_heatmap(rniah)))
    if plad is not None and composite.plad_score is not None:
        axes_html.append(_axis_block("plad", composite.plad_score, _plad_perturb_bars(plad)))

    diag_html = (
        '<ul>' + "".join(f"<li>{_esc(s)}</li>" for s in diag) + '</ul>'
        if diag else ""
    )

    raw = _json.dumps(raw_json or {}, indent=2, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFRACT report — {_esc(model_meta.get("name", model))}</title>
<style>{_CSS}
.composite-card {{ --band-color: {band_color}; }}
</style>
</head>
<body>
<div class="report">
  <header>
    <h1>REFRACT v{_esc(__version__)} — {_esc(model_meta.get("name", model))}</h1>
    <div class="subtitle">Higher is better. Score range 0–100. Generated {_dt.datetime.now().isoformat(timespec="seconds")}.</div>
  </header>

  <section>
    <h2>REFRACT score</h2>
    <div class="composite-card">
      <div class="composite-number">{composite.composite:.2f}</div>
      <div style="flex:1;">
        <div class="composite-band" style="color:{band_color};">{_esc(composite.band)}</div>
        {_bar(composite.composite, composite.band)}
        <div class="composite-summary">{_esc(_BAND_PROSE.get(composite.band, ''))}</div>
      </div>
    </div>
  </section>

  {f'<section><h2>Diagnosis</h2><div class="diagnosis">{diag_html}</div></section>' if diag else ''}

  <section>
    <h2>Per-axis breakdown</h2>
    {''.join(axes_html)}
  </section>

  <section>
    <h2>Run details</h2>
    {_details_block(model_meta, hw_meta, reference_label, candidate_label, env_meta)}
  </section>

  <section>
    <h2>Reproduce</h2>
    <pre class="repro">{_esc(repro)}</pre>
  </section>

  <details>
    <summary>Raw JSON (machine-readable)</summary>
    <pre>{_esc(raw)}</pre>
  </details>

  <footer style="margin-top:32px; padding-top:16px; border-top:1px solid #eee; color:#888; font-size:12px; text-align:center;">
    REFRACT v{_esc(__version__)}.
    What is this? See
    <a href="https://github.com/TheTom/turboquant_plus/tree/main/refract" style="color:#3aa05a;">github.com/TheTom/turboquant_plus/refract</a>
    for documentation, the motivation paper, and how to interpret these scores.
  </footer>
</div>
</body>
</html>
"""
