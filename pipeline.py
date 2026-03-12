#!/usr/bin/env python3
"""
Eco-Mixer Pipeline — Full MASE Toolflow
========================================

This script runs the complete MASE hardware generation toolflow:

  Step 1: Trace model → MaseGraph          (FX tracing)
  Step 2: init_metadata_analysis_pass      (attach MaseMetadata to nodes)
  Step 3: add_common_metadata_analysis_pass (shapes, dtypes, mase_op)

  Step 3.5: mixed_precision_search
        - Receives: MaseGraph with common metadata, accuracy constraint, resource budget
        - Evaluates: per-layer bit-width choices (e.g. linear → 4-bit, relu → 8-bit)
        - Returns: quan_args dict with per-layer config (by="name" or by="type")

        {
            "by": "name",
            "default": {"config": {"name": None}},
            "fc1": {"config": {"name": "fixed", "data_in_width": 4, "weight_width": 4, ...}},
            "relu": {"config": {"name": "fixed", "data_in_width": 8, ...}},
        }

  Step 4: quantize_transform_pass          (fixed-point quantization)
  Step 5: add_hardware_metadata_analysis_pass (RTL module mapping)
  Step 6: emit_verilog_top_transform_pass  (generate top.sv)
  Step 7: emit_internal_rtl_transform_pass (copy component RTL files)
  Step 8: emit_bram_transform_pass         (generate BRAM .dat files)
  Step 9: emit_cocotb_transform_pass       (optional — requires cocotb)

Usage:
    python pipeline.py
"""

import sys
import os
from pathlib import Path
from types import ModuleType

# Ensure Unicode prints work on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

# Optional-dependency stub
# cvxpy is imported at module level in mase/chop/passes/graph/analysis/autosharding/
# but only used inside functions — stub it so the module loads without error.
sys.modules.setdefault("cvxpy", ModuleType("cvxpy"))

# Path setup
# pipeline.py lives at <mase_root>/pipeline.py, so:
#   MASE_ROOT = /home/jl7422/mase
#   MASE_SRC  = /home/jl7422/mase/src  ← needed for "from chop.*" imports
MASE_ROOT = str(Path(__file__).resolve().parent)
MASE_SRC = os.path.join(MASE_ROOT, "src")
for p in [MASE_ROOT, MASE_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

# ── Core graph IR ──────────────────────────────────────────────────────────────
from chop.ir.graph.mase_graph import MaseGraph

# ── Analysis passes ────────────────────────────────────────────────────────────
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

# ── Transform passes ───────────────────────────────────────────────────────────
from chop.passes.graph.transforms.quantize import quantize_transform_pass

# Verilog emit passes (Steps 6-8)
from chop.passes.graph.transforms.verilog.emit_top import (
    emit_verilog_top_transform_pass,
)
from chop.passes.graph.transforms.verilog.emit_internal import (
    emit_internal_rtl_transform_pass,
)
from chop.passes.graph.transforms.verilog.emit_bram import (
    emit_bram_transform_pass,
)

# Step 9 (cocotb testbench) — None when cocotb/mase_cocotb is not installed
from chop.passes.graph.transforms.verilog import emit_cocotb_transform_pass

from chop.tools.logger import set_logging_verbosity

torch.manual_seed(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Definition — matches lab4-hardware.ipynb exactly
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Toy FC model for digit recognition on MNIST (from lab4)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════════════

def step1_trace(model):
    """Step 1: Trace model → MaseGraph."""
    print("\n" + "=" * 70)
    print("STEP 1: Trace model → MaseGraph")
    print("=" * 70)

    mg = MaseGraph(model=model)

    print("FX Graph:")
    mg.fx_graph.print_tabular()
    return mg


def step2_init_metadata(mg):
    """Step 2: Attach empty MaseMetadata to every node."""
    print("\n" + "=" * 70)
    print("STEP 2: init_metadata_analysis_pass")
    print("=" * 70)

    mg, _ = init_metadata_analysis_pass(mg, None)
    print("  ✓ MaseMetadata attached to all nodes")
    return mg


def step3_common_metadata(mg, dummy_in):
    """Step 3: Propagate mase_op, shapes, dtypes via real forward pass."""
    print("\n" + "=" * 70)
    print("STEP 3: add_common_metadata_analysis_pass")
    print("=" * 70)

    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

    print("\nCommon metadata on compute nodes:\n")
    for node in mg.nodes:
        meta = node.meta["mase"]["common"]
        op = meta.get("mase_op")
        if op in (None, "placeholder", "output", "constant"):
            continue
        print(f"  [{node.name}]  mase_op={op}  mase_type={meta.get('mase_type')}")
        for port, info in meta.get("args", {}).items():
            if isinstance(info, dict) and "shape" in info:
                print(f"    arg  {port:12s}: shape={info['shape']}  "
                      f"type={info.get('type')}  precision={info.get('precision')}")
        for port, info in meta.get("results", {}).items():
            if isinstance(info, dict) and "shape" in info:
                print(f"    out  {port:12s}: shape={info['shape']}  "
                      f"type={info.get('type')}  precision={info.get('precision')}")
        print()
    return mg


def step4_quantize(mg, quan_args, fixed_width=8, fixed_frac_width=3):
    """Step 4: Quantize + mandatory precision patch."""
    print("\n" + "=" * 70)
    print("STEP 4: quantize_transform_pass + precision patch")
    print("=" * 70)

    # Apply quantization (replaces nn.Linear → LinearInteger, etc.)
    mg, _ = quantize_transform_pass(mg, quan_args)

    # Report node types after quantization
    print("\nNode types after quantization:")
    report_node_type_analysis_pass(mg)

    # ── MANDATORY PRECISION PATCH ──────────────────────────────────────────
    # quantize_transform_pass updates type/precision for input args (data_in_0,
    # weight, bias) but does NOT update results (data_out_0) for fixed-point.
    # add_hardware_metadata_analysis_pass reads BOTH args and results to produce
    # verilog_param, so results MUST be patched manually.
    precision = [fixed_width, fixed_frac_width]
    for node in mg.fx_graph.nodes:
        for port, info in node.meta["mase"]["common"]["args"].items():
            if isinstance(info, dict):
                info["type"] = "fixed"
                info["precision"] = list(precision)
        for port, info in node.meta["mase"]["common"]["results"].items():
            if isinstance(info, dict):
                info["type"] = "fixed"
                info["precision"] = list(precision)

    print(f"\n  ✓ Precision patch applied: type=fixed, precision={precision}")

    # Spot-check first linear node
    for node in mg.fx_graph.nodes:
        if node.meta["mase"]["common"].get("mase_op") == "linear":
            print(f"\n  Spot-check [{node.name}]:")
            for port, info in node.meta["mase"]["common"]["args"].items():
                if isinstance(info, dict):
                    print(f"    arg {port:12s}: type={info['type']}  "
                          f"precision={info['precision']}")
            break

    return mg


def step5_hardware_metadata(mg, pass_args=None):
    """Step 5: Add hardware metadata (toolchain, module, verilog_param)."""
    print("\n" + "=" * 70)
    print("STEP 5: add_hardware_metadata_analysis_pass")
    print("=" * 70)

    if pass_args is None:
        pass_args = {}

    mg, _ = add_hardware_metadata_analysis_pass(mg, pass_args)

    print(f"\n  {'Node':30s}  {'toolchain':12s}  {'module':25s}  verilog_param")
    print("  " + "-" * 100)
    for node in mg.fx_graph.nodes:
        hw = node.meta["mase"]["hardware"]
        if hw.get("is_implicit"):
            continue
        module = hw.get("module") or ""
        if not module:
            continue
        tc = hw.get("toolchain", "")
        vp = hw.get("verilog_param", {})
        print(f"  {node.name:30s}  {tc:12s}  {module:25s}  {vp}")

    return mg


def steps6to9_emit(mg, project_dir=None):
    """Steps 6-9: RTL emission passes."""
    print("\n" + "=" * 70)
    print("STEPS 6-9: RTL Emission")
    print("=" * 70)

    if project_dir is None:
        # Default: emit inside the repository at <project_root>/top/
        project_dir = Path(__file__).resolve().parent / "top"

    pass_args = {"project_dir": str(project_dir)}

    # Step 6: top.sv
    print("\n  Step 6: emit_verilog_top_transform_pass ...")
    mg, _ = emit_verilog_top_transform_pass(mg, pass_args)
    print(f"    ✓ Generated {project_dir}/hardware/rtl/top.sv")

    # Step 7: copy component .sv files
    print("  Step 7: emit_internal_rtl_transform_pass ...")
    mg, _ = emit_internal_rtl_transform_pass(mg, pass_args)
    print(f"    ✓ RTL dependencies copied")

    # Step 8: BRAM sources
    print("  Step 8: emit_bram_transform_pass ...")
    mg, _ = emit_bram_transform_pass(mg, pass_args)
    print(f"    ✓ BRAM source modules and .dat files generated")

    # Step 9: cocotb testbench (optional — requires cocotb simulator)
    if emit_cocotb_transform_pass is not None:
        print("  Step 9: emit_cocotb_transform_pass ...")
        mg, _ = emit_cocotb_transform_pass(mg, pass_args)
        print(f"    ✓ Cocotb testbench generated")
    else:
        print("  Step 9: skipped (cocotb not installed)")

    # List generated files
    print(f"\n  Generated project tree: {project_dir}")
    for root, dirs, files in os.walk(project_dir):
        level = root.replace(str(project_dir), "").count(os.sep)
        indent = "    " + "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "    " + "  " * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{sub_indent}{f}  ({size} bytes)")

    return mg


def summarise(mg):
    """Print final graph state — common + hardware metadata per node."""
    print("\n" + "=" * 70)
    print("FINAL GRAPH STATE — common + hardware metadata per node")
    print("=" * 70)

    for node in mg.fx_graph.nodes:
        m_common = node.meta["mase"]["common"]
        m_hw = node.meta["mase"]["hardware"]
        op = m_common.get("mase_op", node.op)

        if op in ("placeholder", "output", "constant"):
            continue
        if m_hw.get("is_implicit"):
            continue

        print(f"\n  Node : {node.name}")
        print(f"  op   : {op}  |  mase_type: {m_common.get('mase_type')}")

        for port, info in m_common.get("args", {}).items():
            if isinstance(info, dict) and "shape" in info:
                print(f"    arg  {port:12s}: shape={info['shape']}  "
                      f"type={info.get('type')}  precision={info.get('precision')}")

        if m_hw.get("module"):
            print(f"    hw toolchain : {m_hw['toolchain']}")
            print(f"    hw module    : {m_hw['module']}")
            vp = m_hw.get("verilog_param", {})
            for k, v in vp.items():
                print(f"      {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    model=None,
    dummy_in=None,
    quan_args=None,
    fixed_width=8,
    fixed_frac_width=3,
    project_dir=None,
    run_emit=True,
):
    """Run the complete MASE toolflow pipeline.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to process.
    dummy_in : dict
        Dummy inputs matching model.forward() argument names.
    quan_args : dict
        Quantization config (same format as MASE fixed.toml).
    fixed_width : int
        Total bit width for fixed-point quantization.
    fixed_frac_width : int
        Fractional bit width for fixed-point quantization.
    project_dir : Path, optional
        Output directory for RTL files (default: ~/.mase/top/).
    run_emit : bool
        If True, run Steps 6-9 (RTL emission).

    Returns
    -------
    MaseGraph with all metadata populated.
    """
    set_logging_verbosity("info")

    if model is None:
        model = MLP()
    if dummy_in is None:
        dummy_in = {"x": torch.randn(1, 2, 2)}
    if quan_args is None:
        quan_args = {
            "by": "type",
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "fixed",
                    "data_in_width": fixed_width,
                    "data_in_frac_width": fixed_frac_width,
                    "weight_width": fixed_width,
                    "weight_frac_width": fixed_frac_width,
                    "bias_width": fixed_width,
                    "bias_frac_width": fixed_frac_width,
                }
            },
        }

    print(f"Model: {model.__class__.__name__}")
    print(model)

    # Steps 1-3: Trace + metadata
    mg = step1_trace(model)
    mg = step2_init_metadata(mg)
    mg = step3_common_metadata(mg, dummy_in)

    # Step 4: Quantize + precision patch
    mg = step4_quantize(mg, quan_args, fixed_width, fixed_frac_width)

    # Step 5: Hardware metadata
    mg = step5_hardware_metadata(mg)

    # Steps 6-9: RTL emission
    if run_emit:
        mg = steps6to9_emit(mg, project_dir)

    # Final summary
    try:
        summarise(mg)
    except (KeyError, TypeError):
        # Hardware metadata may not be populated if Step 5 was skipped
        _summarise_common_only(mg)

    print("\n✓ Pipeline complete.")
    return mg


def _summarise_common_only(mg):
    """Fallback summary when hardware metadata is not available."""
    TARGET_OPS = {"linear", "conv1d", "conv2d", "relu", "elu",
                  "sigmoid", "tanh", "gelu", "selu", "leaky_relu",
                  "silu", "softplus", "softsign"}
    print("\n" + "=" * 65)
    print("GRAPH STATE — common metadata (hardware metadata not available)")
    print("=" * 65)
    print(f"  {'node':20s}  {'mase_op':12s}  {'input shape':20s}  output shape")
    print(f"  {'-'*20}  {'-'*12}  {'-'*20}  {'-'*20}")
    for node in mg.nodes:
        meta = node.meta["mase"]["common"]
        if not meta:
            continue
        op = meta.get("mase_op", "")
        if op not in TARGET_OPS:
            continue
        in_shape = str(meta.get("args", {}).get("data_in_0", {}).get("shape", "?"))
        out_shape = str(meta.get("results", {}).get("data_out_0", {}).get("shape", "?"))
        prec = str(meta.get("args", {}).get("data_in_0", {}).get("precision", "?"))
        print(f"  {node.name:20s}  {op:12s}  {in_shape:20s}  {out_shape}")


if __name__ == "__main__":
    mg = run_pipeline()