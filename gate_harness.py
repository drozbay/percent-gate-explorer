"""Percent gate harness.

Runs INSIDE one ComfyUI source tree (cwd = tree root) and reports, for every node that gates on
start_percent / end_percent, which sampler steps the node is active on. Nothing is reimplemented:

  * step sigmas come from the tree's own comfy.samplers.calculate_sigmas
  * bounds come from the tree's own ModelSampling.percent_to_sigma
  * each gate is decided by calling the tree's own node code with stub model objects and observing
    whether the node acted (made its extra model call, reached its control model, changed cfg, ...)

The same file is run against the master tree and the PR tree; the two outputs are then compared.
"""
import argparse
import base64
import json
import os
import random
import re
import struct
import sys
import time
import traceback
import types

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="output directory for per-config json")
ap.add_argument("--label", required=True, help="master | pr")
ap.add_argument("--list-configs", action="store_true", help="print the config list as json and exit")
ap.add_argument("--configs", default="", help="comma separated config ids to run (default: all)")
ap.add_argument("--max-steps", type=int, default=50)
ap.add_argument("--selfcheck", action="store_true", help="only check that every adapter can observe both states")
ap.add_argument("--verify-pairs", type=int, default=300)
ARGS = ap.parse_args()

TREE = os.getcwd()
sys.argv = [sys.argv[0], "--cpu"]          # comfy parses sys.argv at import; never touch the GPU
sys.path.insert(0, TREE)

import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
from unittest.mock import MagicMock

import comfy.model_base
import comfy.model_sampling
import comfy.samplers
import comfy.sampler_helpers
import comfy.controlnet
import comfy.patcher_extension
import comfy.supported_models
import node_helpers
import nodes as core_nodes

PGRID = 1000  # percent widgets move in 0.001 increments


# ----------------------------------------------------------------------------------------------
# model sampling: built exactly the way ComfyUI builds it for a loaded model
# ----------------------------------------------------------------------------------------------
class _Cfg:
    def __init__(self, settings):
        self.sampling_settings = settings


def make_model_sampling(family, shift):
    MT = comfy.model_base.ModelType
    if family == "flow1000":
        return comfy.model_base.model_sampling(_Cfg({"shift": shift, "multiplier": 1000}), MT.FLOW)
    if family == "flow1":
        return comfy.model_base.model_sampling(_Cfg({"shift": shift, "multiplier": 1.0}), MT.FLOW)
    if family == "flux":
        return comfy.model_base.model_sampling(_Cfg({"shift": shift}), MT.FLUX)
    if family == "flow_av":
        return comfy.model_base.model_sampling(_Cfg({"shift": shift, "audio_shift": 3.0}), MT.FLOW_AV)
    if family == "eps":
        return comfy.model_base.model_sampling(_Cfg({}), MT.EPS)
    if family == "sensenova":
        import comfy.ldm.sensenova.sampling as sn
        return sn.SenseNovaModelSampling(_Cfg({"shift": shift, "noise_scale": 1.0}))
    raise ValueError(family)


FAMILY_LABEL = {
    "flow1000": "ModelSamplingDiscreteFlow, multiplier 1000",
    "flow1": "ModelSamplingDiscreteFlow, multiplier 1",
    "flux": "ModelSamplingFlux",
    "flow_av": "ModelSamplingAV (MiniMax H3)",
    "eps": "ModelSamplingDiscrete (eps)",
    "sensenova": "SenseNovaModelSampling",
}


def discover_models():
    """Read every supported model's real sampling_settings and model type from the tree's source."""
    import inspect
    MT = comfy.model_base.ModelType
    out = []
    for cls in comfy.supported_models.models:
        try:
            settings = dict(getattr(cls, "sampling_settings", {}) or {})
            src = inspect.getsource(cls.get_model)
            mt = None
            m = re.search(r"model_type\s*=\s*model_base\.ModelType\.(\w+)", src)
            if m:
                mt = getattr(MT, m.group(1))
            if mt is None:
                m = re.search(r"model_base\.(\w+)\(", src)
                if m:
                    base = getattr(comfy.model_base, m.group(1))
                    sig = inspect.signature(base.__init__)
                    if "model_type" in sig.parameters and sig.parameters["model_type"].default is not inspect._empty:
                        mt = sig.parameters["model_type"].default
            if mt is None:
                continue
            if cls.__name__.startswith("SenseNova"):
                fam = "sensenova"
            elif mt == MT.FLOW:
                fam = "flow1" if float(settings.get("multiplier", 1000)) == 1.0 else "flow1000"
                if float(settings.get("multiplier", 1000)) not in (1.0, 1000.0):
                    continue
            elif mt == MT.FLUX:
                fam = "flux"
            elif mt == MT.FLOW_AV:
                fam = "flow_av"
            elif mt == MT.EPS:
                if settings:
                    continue  # non default beta schedules: not covered
                fam = "eps"
            else:
                continue
            default_shift = {"flux": 1.15}.get(fam, 1.0)
            shift = float(settings.get("shift", default_shift)) if fam != "eps" else 0.0
            out.append({"name": cls.__name__, "family": fam, "shift": shift})
        except Exception:
            continue
    seen, uniq = set(), []
    for m in out:
        if m["name"] not in seen:
            seen.add(m["name"])
            uniq.append(m)
    return uniq


def config_id(family, shift):
    return f"{family}_{shift:g}".replace(".", "p")


def all_configs():
    models = discover_models()
    cfgs = {}
    for m in models:
        cfgs[config_id(m["family"], m["shift"])] = (m["family"], m["shift"])
    for family in ("flow1000", "flow1", "flow_av", "sensenova"):     # shift overrides set with the model sampling nodes
        for s in [0.5] + list(range(1, 16)):
            cfgs[config_id(family, float(s))] = (family, float(s))
    for s in (0.5, 0.8, 1.15, 2.05, 2.37):                   # ModelSamplingFlux node range
        cfgs[config_id("flux", s)] = ("flux", s)
    return models, dict(sorted(cfgs.items()))


# ----------------------------------------------------------------------------------------------
# stubs
# ----------------------------------------------------------------------------------------------
class Reached(Exception):
    """Raised by a tripwire: the node got past its gate and reached for the model."""


class Tripwire:
    def __getattr__(self, name):
        raise Reached(name)


class Rec:
    """Stands in for a ModelPatcher. Hands out the real model_sampling and records what the node installs."""

    def __init__(self, ms):
        self.ms = ms
        self.installed = []
        self.model_options = {"transformer_options": {}}
        self.objects = {}

    def clone(self):
        return self

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.ms
        return self.objects.setdefault(name, MagicMock())

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def rec(*a, **k):
            self.installed.append((name, a, k))
        return rec

    def last(self, name):
        hits = [x for x in self.installed if x[0] == name]
        if not hits:
            raise RuntimeError(f"node did not call {name}")
        return hits[-1]


class FakeBaseModel:
    def __init__(self, ms):
        self.model_sampling = ms
        self.current_patcher = None


X4 = torch.zeros(1, 4, 8, 8)
COND = [[torch.zeros(1, 1, 4), {}]]
CALLS = []


def _calc_cond_batch_stub(model, conds, x, sigma, model_options):
    CALLS.append(1)
    return [torch.zeros_like(x) for _ in conds]


comfy.samplers.calc_cond_batch = _calc_cond_batch_stub   # the expensive model call every guidance gate protects


def node_out(out):
    for attr in ("args", "result"):
        v = getattr(out, attr, None)
        if v:
            return v
    return out


# ----------------------------------------------------------------------------------------------
# adapters: build(ms, start, end) -> probe(sigma float32 tensor of shape [1]) -> bool
# ----------------------------------------------------------------------------------------------
SITES = []


def site(id, label, file, pattern, native=None, sigma_source="own", requires_end_gt_start=False, note=""):
    def deco(fn):
        SITES.append({"id": id, "label": label, "file": file, "pattern": pattern, "native": native or ["*"],
                      "sigma_source": sigma_source, "requires_end_gt_start": requires_end_gt_start, "note": note, "build": fn})
        return fn
    return deco


def _post_cfg_probe(fn):
    def probe(sigma):
        CALLS.clear()
        fn({"model": None, "cond_denoised": X4, "cond": [], "uncond": [], "denoised": X4, "sigma": sigma,
            "input": X4, "model_options": {"transformer_options": {}}})
        return len(CALLS) > 0
    return probe


@site("cond_range", "ConditioningSetTimestepRange", "comfy/samplers.py", r"timestep_in\[0\] <=? timestep_end")
def b_cond_range(ms, start, end):
    c = core_nodes.ConditioningSetTimestepRange().set_range(COND, start, end)[0]
    conds = comfy.sampler_helpers.convert_cond(c)
    comfy.samplers.calculate_start_end_timesteps(FakeBaseModel(ms), conds)
    cond = conds[0]
    return lambda sigma: comfy.samplers.get_area_and_mult(cond, X4, sigma) is not None


def _control_probe(cn, ms, start, end):
    pos, _ = core_nodes.ControlNetApplyAdvanced().apply_controlnet(COND, COND, cn, torch.zeros(1, 8, 8, 3), 1.0, start, end)
    conds = comfy.sampler_helpers.convert_cond(pos)
    comfy.samplers.pre_run_control(FakeBaseModel(ms), conds)
    ctrl = conds[0]["control"]

    def probe(sigma):
        try:
            r = ctrl.get_control(X4, sigma, {}, 1, {})
        except Reached:
            return True
        if r is not None:
            raise RuntimeError("control returned a value without reaching the model")
        return False
    return probe


@site("controlnet", "Apply ControlNet", "comfy/controlnet.py", r"class ControlNet\(.*?(if t\[0\] > self\.timestep_range\[0\] or t\[0\] <=? self\.timestep_range\[1\])",
      native=["eps", "SD3", "Flux", "QwenImage", "HunyuanDiT", "ZImage"])
def b_controlnet(ms, start, end):
    cn = comfy.controlnet.ControlNet(None)
    cn.control_model = Tripwire()
    cn.control_model_wrapped = None
    return _control_probe(cn, ms, start, end)


@site("t2i", "Apply ControlNet (T2I-Adapter)", "comfy/controlnet.py", r"class T2IAdapter\(.*?(if t\[0\] > self\.timestep_range\[0\] or t\[0\] <=? self\.timestep_range\[1\])",
      native=["eps"])
def b_t2i(ms, start, end):
    ad = comfy.controlnet.T2IAdapter(Tripwire(), 3, 8, "nearest-exact", device=torch.device("cpu"))
    orig_copy = ad.copy
    return _control_probe(ad, ms, start, end) if orig_copy else None


@site("slg", "SkipLayerGuidanceDiT", "comfy_extras/nodes_slg.py", r"if scale > 0 and sigma_ >=? sigma_end and sigma_ <= sigma_start",
      native=["flow", "flux"])
def b_slg(ms, start, end):
    import comfy_extras.nodes_slg as m
    r = Rec(ms)
    m.SkipLayerGuidanceDiT.execute(r, scale=3.0, start_percent=start, end_percent=end, double_layers="7", single_layers="", rescaling_scale=0.0)
    return _post_cfg_probe(r.last("set_model_sampler_post_cfg_function")[1][0])


@site("slg_simple", "SkipLayerGuidanceDiTSimple", "comfy_extras/nodes_slg.py", r"if sigma_ >=? sigma_end and sigma_ <= sigma_start and uncond is not None",
      native=["flow", "flux"])
def b_slg_simple(ms, start, end):
    import comfy_extras.nodes_slg as m
    r = Rec(ms)
    m.SkipLayerGuidanceDiTSimple.execute(r, start_percent=start, end_percent=end, double_layers="7", single_layers="")
    fn = r.last("set_model_sampler_calc_cond_batch_function")[1][0]

    def probe(sigma):
        CALLS.clear()
        fn({"input": X4, "model": None, "conds": [[], []], "sigma": sigma, "model_options": {"transformer_options": {}}})
        if len(CALLS) not in (1, 2):
            raise RuntimeError(f"unexpected call count {len(CALLS)}")
        return len(CALLS) == 2      # gated on: separate cond / skipped-layer uncond passes
    return probe


@site("ltx_stg", "LTXV Spatio-Temporal Guidance (STG)", "comfy_extras/nodes_lt.py", r"class LTXVSpatioTemporalGuidance.*?(if sigma_ > sigma_start or sigma_ <=? sigma_end)",
      native=["LTXV", "LTXAV"])
def b_stg(ms, start, end):
    import comfy_extras.nodes_lt as m
    r = Rec(ms)
    m.LTXVSpatioTemporalGuidance.execute(r, scale=6.0, blocks="14", start_percent=start, end_percent=end)
    return _post_cfg_probe(r.last("set_model_sampler_post_cfg_function")[1][0])


@site("ltx_modality", "LTXV Modality Guidance", "comfy_extras/nodes_lt.py", r"class LTXVModalityGuidance.*?(if sigma_ > sigma_start or sigma_ <=? sigma_end)",
      native=["LTXV", "LTXAV"])
def b_modality(ms, start, end):
    import comfy_extras.nodes_lt as m
    r = Rec(ms)
    m.LTXVModalityGuidance.execute(r, modality_scale=3.0, start_percent=start, end_percent=end)
    return _post_cfg_probe(r.last("set_model_sampler_post_cfg_function")[1][0])


@site("ltx_ref_audio", "LTXV Reference Audio (identity guidance)", "comfy_extras/nodes_lt.py", r"class LTXVReferenceAudio.*?(if sigma_ > sigma_start or sigma_ <=? sigma_end)",
      native=["LTXV", "LTXAV"])
def b_ref_audio(ms, start, end):
    import comfy_extras.nodes_lt as m
    r = Rec(ms)
    vae = types.SimpleNamespace(audio_sample_rate=44100, encode=lambda w: torch.zeros(1, 8, 4, 16))
    audio = {"waveform": torch.zeros(1, 1, 64), "sample_rate": 44100}
    m.LTXVReferenceAudio.execute(r, COND, COND, audio, vae, 3.0, start, end)
    return _post_cfg_probe(r.last("set_model_sampler_post_cfg_function")[1][0])


@site("cfg_override", "CFG Override", "comfy_extras/nodes_custom_sampler.py", r"if not \(sigma_lo <=? sigma <= sigma_hi\)")
def b_cfg_override(ms, start, end):
    import comfy_extras.nodes_custom_sampler as m
    r = Rec(ms)
    m.CFGOverride.execute(r, 3.0, start, end)
    fn = r.last("add_wrapper")[1][1]

    class Exec:
        def __init__(self):
            self.class_obj = types.SimpleNamespace(cfg=1.0)
            self.seen = None

        def __call__(self, *a, **k):
            self.seen = self.class_obj.cfg

    def probe(sigma):
        ex = Exec()
        fn(ex, X4, sigma, {}, 0)
        if ex.seen not in (1.0, 3.0):
            raise RuntimeError("cfg not observed")
        return ex.seen == 3.0
    return probe


@site("sa_solver", "SamplerSASolver (SDE window)", "comfy/k_diffusion/sa_solver.py", r"return eta if start_sigma >= sigma >=? end_sigma else 0\.0",
      note="Each cell is the window's verdict on that step's sigma. The sampler asks about the sigma it is stepping to, so the noise itself lands one step earlier.")
def b_sa_solver(ms, start, end):
    import comfy_extras.nodes_custom_sampler as m
    r = Rec(ms)
    out = m.SamplerSASolver.execute(r, 1.0, start, end, 1.0, 3, 4, False, False)
    sampler = node_out(out)[0]
    tau = sampler.extra_options["tau_func"]
    return lambda sigma: tau(sigma[0]) > 0


@site("downscale", "PatchModelAddDownscale (Kohya Deep Shrink)", "comfy_extras/nodes_model_downscale.py", r"if sigma <= sigma_start and sigma >=? sigma_end",
      native=["eps"])
def b_downscale(ms, start, end):
    import comfy_extras.nodes_model_downscale as m
    r = Rec(ms)
    m.PatchModelAddDownscale.execute(r, 3, 2.0, start, end, True, "bicubic", "bicubic")
    fn = r.last("set_model_input_block_patch_after_skip")[1][0]
    h = torch.zeros(1, 4, 8, 8)
    return lambda sigma: fn(h, {"block": ("input", 3), "sigmas": sigma}).shape != h.shape


_SPARSE = []


@site("sparse_attn", "Block Sparse Attention", "comfy_extras/nodes_sparse_attention.py", r"if sigma > self\.sigma_start or sigma <=? self\.sigma_end",
      native=["MiniMaxH3"])
def b_sparse(ms, start, end):
    import comfy_extras.nodes_sparse_attention as m
    if not getattr(m.SparseAttnPatch, "_pge_wrapped", False):
        orig = m.SparseAttnPatch.__init__

        def init(self, *a, **k):
            orig(self, *a, **k)
            _SPARSE.append(self)
        m.SparseAttnPatch.__init__ = init
        m.SparseAttnPatch._pge_wrapped = True
    _SPARSE.clear()
    r = Rec(ms)
    m.BlockSparseAttention.execute(r, {"selection": "sol-attn"}, start, end)
    patch = _SPARSE[-1]

    def probe(sigma):
        reason = patch.dense_reason({"sigmas": sigma}, 10 ** 7, 0)
        if reason is None:
            return True
        if "outside the start/end window" in reason:
            return False
        raise RuntimeError(reason)
    return probe


@site("h3_control", "Apply MiniMax H3 Fun ControlNet", "comfy_extras/nodes_minimax_h3.py", r"self\.active = self\.sigma_end <=? sigma <= self\.sigma_start",
      native=["MiniMaxH3"])
def b_h3_control(ms, start, end):
    import comfy_extras.nodes_minimax_h3 as m
    r = Rec(ms)
    mp = MagicMock()
    mp.model.injection_layers = [0]
    m.MiniMaxH3FunControlNetApply.execute(r, mp, MagicMock(), 1.0, start, end, control_video=torch.zeros(1, 8, 8, 3))
    wrapper = r.last("add_wrapper")[1][1]
    patch = wrapper.__self__
    patch.prepare_control_latent = lambda shape: None      # VAE encode of the control video; after the gate

    def probe(sigma):
        wrapper(lambda *a, **k: None, X4, sigma * 1000.0, None, {"sigmas": sigma})
        return bool(patch.active)
    return probe


@site("uni3c", "Apply Uni3C ControlNet (Wan)", "comfy_extras/nodes_model_patch.py", r"class WanUni3CCnetPatch.*?(if sigma > self\.sigma_start or sigma <=? self\.sigma_end)",
      native=["WAN"])
def b_uni3c(ms, start, end):
    import comfy_extras.nodes_model_patch as m
    import comfy.ldm.wan.uni3c as u
    r = Rec(ms)
    r.objects["diffusion_model"] = types.SimpleNamespace(dim=5120)
    mp = MagicMock()
    mp.model = MagicMock(spec=u.WanUni3CControlnet)
    blk = MagicMock()
    blk.norm1.linear.in_features = 5120
    mp.model.controlnet_blocks = [blk]
    mp.model.num_layers = 0
    m.WanUni3CControlnetApply().apply_patch(r, mp, MagicMock(), torch.zeros(1, 8, 8, 3), 1.0, start, end)
    patch = r.last("set_model_double_block_patch")[1][0]

    def probe(sigma):
        try:
            patch({"img": torch.zeros(1, 4, 8), "block_index": 0, "transformer_options": {"sigmas": sigma}, "x": Tripwire(), "vec": None})
        except Reached:
            return True
        return False
    return probe


@site("lllite", "Apply Anima LLLite", "comfy/ldm/anima/lllite.py", r"if not self\.sigma_end <=? sigma <= self\.sigma_start",
      native=["Anima"])
def b_lllite(ms, start, end):
    import comfy_extras.nodes_model_patch as m
    r = Rec(ms)
    mp = MagicMock()
    mp.model.cond_in_channels = 3
    m.AnimaLLLiteApply().apply_patch(r, mp, torch.zeros(1, 8, 8, 3), 1.0, start, end)
    patch = r.last("set_model_post_input_patch")[1][0]
    x = torch.zeros(1, 4, 2, 8, 8)       # T=2: the node's own first check after the gate rejects it

    def probe(sigma):
        try:
            patch({"x": x, "transformer_options": {"sigmas": sigma}})
        except ValueError as e:
            if "only supports T=1" in str(e):
                return True
            raise
        return False
    return probe


@site("hidream_o1", "HiDream O1 Patch Seam Smoothing", "comfy_extras/nodes_hidream_o1.py", r"if not \(end_t <=? t <= start_t\)",
      native=["HiDreamO1"], requires_end_gt_start=True, note="Compares in timestep units, after the model's sigma to timestep conversion.")
def b_hidream(ms, start, end):
    import comfy_extras.nodes_hidream_o1 as m
    cls = m.HiDreamO1PatchSeamSmoothing
    pattern = sorted({k[0] for k in cls.SHIFTS_BY_PATTERN if k[1] == 2})[0]
    r = Rec(ms)
    cls.execute(model=r, start_percent=start, end_percent=end, pattern=pattern, passes="2", blend="average", strength=1.0)
    hits = [x for x in r.installed if x[0] == "add_wrapper_with_key"]
    if not hits:
        return lambda sigma: False        # node returned the model untouched
    fn = hits[-1][1][2]
    x = torch.zeros(1, 1, 64, 64)

    class Exec:
        n = 0

        def __call__(self, *a, **k):
            self.n += 1
            return torch.zeros(1, 1, 64, 64)

    def probe(sigma):
        ex = Exec()
        t = ms.timestep(sigma).float()   # what BaseModel._apply_model hands the diffusion model
        fn(ex, x, t)
        return ex.n > 1
    return probe


def _cache_probe(holder_cls, ms, start, end):
    import inspect
    n = len(inspect.signature(holder_cls.__init__).parameters) - 1
    base = [0.2, start, end, 8, False, False]
    h = holder_cls(*base[:n]).prepare_timesteps(ms)
    return lambda sigma: bool(h.has_started(sigma)) and not bool(h.is_past_end_timestep(sigma)) if hasattr(h, "has_started") \
        else bool(h.should_do_easycache(sigma)) and not bool(h.is_past_end_timestep(sigma))


@site("easycache", "EasyCache", "comfy_extras/nodes_easycache.py", r"class EasyCacheHolder.*?(return not \(timestep\[0\] > self\.end_t\)\.item\(\))",
      note="Gate code is identical on both sides; only the bound it compares against can differ.")
def b_easycache(ms, start, end):
    import comfy_extras.nodes_easycache as m
    return _cache_probe(m.EasyCacheHolder, ms, start, end)


@site("lazycache", "LazyCache", "comfy_extras/nodes_easycache.py", r"class LazyCacheHolder.*?(return not \(timestep\[0\] > self\.end_t\)\.item\(\))",
      note="Gate code is identical on both sides; only the bound it compares against can differ.")
def b_lazycache(ms, start, end):
    import comfy_extras.nodes_easycache as m
    return _cache_probe(m.LazyCacheHolder, ms, start, end)


# ----------------------------------------------------------------------------------------------
# source line extraction (shown on the page next to each node)
# ----------------------------------------------------------------------------------------------
def source_line(s):
    path = os.path.join(TREE, s["file"])
    text = open(path, encoding="utf-8").read()
    m = re.search(s["pattern"], text, re.S)
    if not m:
        return None
    pos = m.start(1) if m.groups() else m.start()
    line_no = text.count("\n", 0, pos) + 1
    return {"file": s["file"], "line": line_no, "text": text.splitlines()[line_no - 1].strip()}


# ----------------------------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------------------------
def f32key(v):
    return struct.pack("<f", v)


def sigma_table(ms, max_steps):
    schedules, values, unavailable = {}, {}, {}
    for name in comfy.samplers.SCHEDULER_NAMES:
        per = {}
        for steps in range(1, max_steps + 1):
            try:
                sig = comfy.samplers.calculate_sigmas(ms, name, steps)
                if sig.dtype != torch.float32 or not torch.isfinite(sig).all() or len(sig) != steps + 1:
                    raise RuntimeError("scheduler did not return steps + 1 finite float32 sigmas")
                vals = [float(v) for v in sig]
                if any(b > a for a, b in zip(vals, vals[1:])):
                    raise RuntimeError("sigmas are not descending")
            except Exception as e:
                unavailable.setdefault(name, {})[str(steps)] = f"{type(e).__name__}: {e}"[:120]
                per[steps] = None
                continue
            for v in vals:
                if v > 0.0:
                    values[f32key(v)] = v
            per[steps] = vals
        if any(v is not None for v in per.values()):
            schedules[name] = per
    table = sorted(values.values())
    index = {f32key(v): i for i, v in enumerate(table)}
    sched_idx = {n: {str(s): ([index[f32key(v)] if v > 0.0 else -1 for v in vals] if vals is not None else None)
                     for s, vals in per.items()} for n, per in schedules.items()}
    return table, sched_idx, unavailable


class Counter:
    builds = 0
    probes = 0


def measure_site(s, ms, table, tensors):
    n = len(table)

    def first_index(probe, want):
        """Smallest j with probe(sigma_j) == want, assuming the answer flips once along ascending sigma."""
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            Counter.probes += 1
            if probe(tensors[mid]) == want:
                hi = mid
            else:
                lo = mid + 1
        return lo

    k_start, k_end = [], []
    for p in range(PGRID + 1):
        pct = p / PGRID
        Counter.builds += 2
        k_start.append(first_index(s["build"](ms, pct, 1.0), False))   # active on sigmas below the start bound
        k_end.append(first_index(s["build"](ms, 0.0, pct), True))      # active on sigmas above the end bound
    return k_start, k_end


def predict(s, k_start, k_end, ps, pe, j):
    if s["requires_end_gt_start"] and pe <= ps:
        return False
    return j < k_start[ps] and j >= k_end[pe]


def verify_site(s, ms, table, tensors, k_start, k_end, sched_idx, rng, pairs):
    """Re-run the real node for full (start, end) pairs and compare against what the page will compute."""
    n = len(table)
    grid = []
    for steps in (4, 5, 8, 10, 20):                                   # percents that land exactly on steps
        for a in range(steps + 1):
            for b in range(steps + 1):
                grid.append((round(a * PGRID / steps), round(b * PGRID / steps)))
    rng.shuffle(grid)
    cases = grid[:pairs // 2]
    while len(cases) < pairs:
        cases.append((rng.randint(0, PGRID), rng.randint(0, PGRID)))
    checked = mismatches = 0
    first = None
    for ps, pe in cases:
        probe = s["build"](ms, ps / PGRID, pe / PGRID)
        js = {0, n - 1, rng.randrange(n), rng.randrange(n)}
        for k in (k_start[ps], k_end[pe]):
            for d in (-2, -1, 0, 1):
                if 0 <= k + d < n:
                    js.add(k + d)
        for j in js:
            got = bool(probe(tensors[j]))
            exp = predict(s, k_start, k_end, ps, pe, j)
            checked += 1
            if got != exp:
                mismatches += 1
                if first is None:
                    first = {"start": ps / PGRID, "end": pe / PGRID, "sigma": table[j], "node": got, "page": exp}
    return checked, mismatches, first


def run_config(cid, family, shift, out_dir):
    t0 = time.time()
    ms = make_model_sampling(family, shift)
    table, sched_idx, unavailable = sigma_table(ms, ARGS.max_steps)
    tensors = [torch.tensor([v], dtype=torch.float32) for v in table]
    for v, t in zip(table, tensors):
        assert float(t[0]) == v
    bounds = [ms.percent_to_sigma(p / PGRID) for p in range(PGRID + 1)]
    rng = random.Random(f"{cid}")
    sites = {}
    for s in SITES:
        Counter.builds = Counter.probes = 0
        try:
            k_start, k_end = measure_site(s, ms, table, tensors)
            checked, mismatches, first = verify_site(s, ms, table, tensors, k_start, k_end, sched_idx, rng, ARGS.verify_pairs)
            sites[s["id"]] = {"ok": True, "k_start": k_start, "k_end": k_end, "verified": checked, "mismatches": mismatches,
                              "first_mismatch": first, "node_runs": Counter.builds, "gate_calls": Counter.probes + checked}
        except Exception as e:
            sites[s["id"]] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    out = {
        "id": cid, "family": family, "shift": shift, "label": ARGS.label,
        "model_sampling_class": ", ".join(c.__name__ for c in type(ms).__mro__[1:3]),
        "sigmas_f32": base64.b64encode(struct.pack(f"<{len(table)}f", *table)).decode(),
        "n_sigmas": len(table), "schedules": sched_idx, "schedulers_unavailable": unavailable,
        "bounds": [repr(float(b)) for b in bounds],
        "sites": sites, "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"{cid}.json"), "w") as f:
        json.dump(out, f)
    bad = {k: v.get("error") or f"{v['mismatches']} mismatches" for k, v in sites.items() if not v["ok"] or v["mismatches"]}
    print(f"[{ARGS.label}] {cid}: {len(table)} sigmas, {out['seconds']}s" + (f"  ISSUES {bad}" if bad else ""), flush=True)


def selfcheck():
    """Every adapter must be able to observe both states, otherwise its rows mean nothing."""
    ok = True
    for family, shift in (("flow1000", 3.0), ("flux", 2.37), ("eps", 0.0)):
        ms = make_model_sampling(family, shift)
        sig = comfy.samplers.calculate_sigmas(ms, "simple", 8)
        mid = sig[4].reshape(1)
        for s in SITES:
            try:
                on = s["build"](ms, 0.0, 1.0)(mid)
                off_late = s["build"](ms, 0.9, 1.0)(mid)
                off_early = s["build"](ms, 0.0, 0.1)(mid)
                good = on is True and off_late is False and off_early is False
                print(f"  {family:9s} {s['id']:14s} open={on} late_window={off_late} early_window={off_early} {'OK' if good else 'BAD'}")
                ok &= good
            except Exception as e:
                print(f"  {family:9s} {s['id']:14s} n/a: {type(e).__name__}: {str(e)[:90]}")
    for s in SITES:
        print(f"  source {s['id']:14s} {source_line(s)}")
    return ok


if __name__ == "__main__":
    models, cfgs = all_configs()
    if ARGS.list_configs:
        print(json.dumps({"configs": {k: list(v) for k, v in cfgs.items()}, "models": models}))
        sys.exit(0)
    if ARGS.selfcheck:
        sys.exit(0 if selfcheck() else 1)
    os.makedirs(ARGS.out, exist_ok=True)
    meta = {
        "label": ARGS.label, "commit": open(os.path.join(TREE, "COMMIT")).read().strip() if os.path.exists(os.path.join(TREE, "COMMIT")) else None,
        "torch": torch.__version__, "python": sys.version.split()[0], "max_steps": ARGS.max_steps,
        "schedulers": list(comfy.samplers.SCHEDULER_NAMES), "models": models, "family_label": FAMILY_LABEL,
        "sites": [{k: v for k, v in s.items() if k not in ("build", "pattern")} | {"source": source_line(s)} for s in SITES],
    }
    with open(os.path.join(ARGS.out, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    want = [c for c in ARGS.configs.split(",") if c] or list(cfgs)
    for cid in want:
        family, shift = cfgs[cid]
        try:
            run_config(cid, family, shift, ARGS.out)
        except Exception:
            print(f"[{ARGS.label}] {cid}: FAILED\n{traceback.format_exc()}", flush=True)
