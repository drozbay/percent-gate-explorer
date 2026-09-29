# Percent Gate Explorer

Shows which sampler steps a ComfyUI node is active on for any `start_percent`, `end_percent`,
scheduler and step count, on ComfyUI master next to
[Comfy-Org/ComfyUI#16156](https://github.com/Comfy-Org/ComfyUI/pull/16156).

The page is static. Its numbers are measured by running ComfyUI's own code from the two commits.

## How a result is produced

`gate_harness.py` runs inside a ComfyUI source tree. The same file runs in the master tree and in
the PR tree.

| Value | Where it comes from |
|---|---|
| Step sigmas | `comfy.samplers.calculate_sigmas` |
| Model sampling object | `comfy.model_base.model_sampling`, with each model's own `sampling_settings` |
| Start and end bounds | that object's `percent_to_sigma` |
| Active or inactive | the node's own code, called with a float32 sigma |

"Active" is observed, not computed. The node is set up with stub model objects, then called. The
harness watches for the thing the gate protects.

| Node | What counts as active |
|---|---|
| ConditioningSetTimestepRange | `get_area_and_mult` returns the conditioning |
| Apply ControlNet, T2I-Adapter | `get_control` gets past the gate and reaches for the control model |
| SkipLayerGuidanceDiT, LTXV STG, Modality, Reference Audio | the node makes its extra `calc_cond_batch` call |
| SkipLayerGuidanceDiTSimple | the node splits cond and uncond into two calls |
| CFG Override | the guider's cfg is the override value during the call |
| SamplerSASolver | `tau_func` returns a non zero value for the sigma |
| PatchModelAddDownscale | the input block patch changes the tensor size |
| Block Sparse Attention | `dense_reason` does not report the window |
| Apply MiniMax H3 Fun ControlNet | the patch sets `active` |
| Apply Uni3C ControlNet | the patch reads the latent |
| Apply Anima LLLite | the patch reaches its first check after the gate |
| HiDream O1 Patch Seam Smoothing | the wrapper runs its extra shifted passes |
| EasyCache, LazyCache | `has_started` is true and `is_past_end_timestep` is false |

Percent inputs move in steps of 0.001, so all 1001 values are measured for start and for end, for
every sigma any scheduler produces up to 50 steps. After measuring, each node is set up again with
full start and end pairs and its answer is compared with the lookup the page uses. The build fails
to publish a row that does not match.

## Build locally

Needs a Python environment that can import ComfyUI, and a ComfyUI checkout that contains both commits.

```bash
python prepare_trees.py --repo /path/to/ComfyUI --pr-sha b29b3230defe9687fde5ece9eb53c38944925894
python build_site.py --jobs 6
python -m http.server 8765 --directory site
```

`--single-file` also writes `site/single.html` with all data inlined.

## Publish

Push this folder to a repository, then set Pages to "GitHub Actions" in the repository settings.
`.github/workflows/pages.yml` fetches the two pinned commits from Comfy-Org/ComfyUI, measures them,
and deploys the page. The page links to the run log that built it.

Update `PR_SHA` and `BASE_SHA` in the workflow when the pull request changes.

## Limits

- One model call per step, at that step's sigma. Samplers that evaluate between steps also call the
  nodes at those sigmas.
- Hook keyframes and the SCAIL pose range are not shown.
- Steps are numbered from 1.
- SamplerSASolver is drawn by the sigma its window is asked about. The sampler asks about the sigma it is
  stepping to, so the noise lands one step earlier than the row shows.
