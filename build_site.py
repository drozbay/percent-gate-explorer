"""Build the percent gate explorer site.

1. Runs gate_harness.py inside the master tree and inside the PR tree (the same file, unmodified, in both).
2. Checks that both trees produce identical step sigmas, merges the two measurements, and fails if any
   spot check disagreed with the lookup the page uses.
3. Writes site/index.html plus one data file per model sampling config.

Usage:
  python build_site.py --master-tree trees/master --pr-tree trees/pr --jobs 6
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "gate_harness.py")

ap = argparse.ArgumentParser()
ap.add_argument("--master-tree", default=os.path.join(HERE, "trees", "master"))
ap.add_argument("--pr-tree", default=os.path.join(HERE, "trees", "pr"))
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--jobs", type=int, default=4, help="parallel workers per tree")
ap.add_argument("--work", default=os.path.join(HERE, "out"))
ap.add_argument("--site", default=os.path.join(HERE, "site"))
ap.add_argument("--repo", default="Comfy-Org/ComfyUI")
ap.add_argument("--pr-number", default="16156")
ap.add_argument("--tool-url", default="", help="public URL of this tool's source, shown on the page")
ap.add_argument("--build-url", default="", help="public URL of the build log that produced the page")
ap.add_argument("--skip-measure", action="store_true")
ap.add_argument("--max-steps", type=int, default=50)
ap.add_argument("--single-file", action="store_true", help="also write site/single.html with all data inlined")
A = ap.parse_args()

TREES = {"master": os.path.abspath(A.master_tree), "pr": os.path.abspath(A.pr_tree)}


def run_harness(side, extra, capture=False, log=None):
    cmd = [A.python, "-s", HARNESS, "--out", os.path.join(A.work, side), "--label", side, "--max-steps", str(A.max_steps)] + extra
    if capture:
        return subprocess.run(cmd, cwd=TREES[side], capture_output=True, text=True)
    return subprocess.Popen(cmd, cwd=TREES[side], stdout=log, stderr=subprocess.STDOUT)


def measure():
    r = run_harness("pr", ["--list-configs"], capture=True)
    if r.returncode != 0:
        sys.exit("could not list configs:\n" + r.stderr[-3000:])
    ids = list(json.loads(r.stdout.strip().splitlines()[-1])["configs"])
    os.makedirs(os.path.join(A.work, "logs"), exist_ok=True)
    procs = []
    for side in TREES:
        os.makedirs(os.path.join(A.work, side), exist_ok=True)
        for s in range(A.jobs):
            shard = ids[s::A.jobs]
            if not shard:
                continue
            log = open(os.path.join(A.work, "logs", f"{side}_{s}.log"), "w")
            procs.append((side, s, run_harness(side, ["--configs", ",".join(shard)], log=log), log))
    failed = False
    for side, s, p, log in procs:
        p.wait()
        log.close()
        if p.returncode != 0:
            failed = True
            print(f"worker {side}/{s} exited with {p.returncode}")
    if failed:
        sys.exit("measurement failed; see logs")
    return ids


def b64(fmt, values):
    return base64.b64encode(struct.pack(f"<{len(values)}{fmt}", *values)).decode()


def merge():
    meta = {side: json.load(open(os.path.join(A.work, side, "_meta.json"))) for side in TREES}
    ids = sorted(f[:-5] for f in os.listdir(os.path.join(A.work, "pr")) if f.endswith(".json") and not f.startswith("_"))
    os.makedirs(os.path.join(A.site, "data"), exist_ok=True)
    totals = {side: {"node_runs": 0, "gate_calls": 0, "verified": 0, "mismatches": 0} for side in TREES}
    configs, problems, payloads, excluded = [], [], {}, []
    max_steps = meta["pr"]["max_steps"]
    for cid in ids:
        d = {side: json.load(open(os.path.join(A.work, side, f"{cid}.json"))) for side in TREES}
        m, p = d["master"], d["pr"]
        if m["sigmas_f32"] != p["sigmas_f32"] or m["schedules"] != p["schedules"]:
            problems.append(f"{cid}: step sigmas differ between master and PR")
            continue
        sched = {}
        for name, per in p["schedules"].items():
            flat = []
            for steps in range(1, max_steps + 1):
                idx = per[str(steps)]
                if idx is None:
                    flat.extend([65534] * (steps + 1))      # scheduler cannot produce this step count
                else:
                    flat.extend(65535 if j < 0 else j for j in idx)
            sched[name] = b64("H", flat)
        sides = {}
        for side in TREES:
            classes, index, site_class, errors = [], {}, {}, {}
            for sid, s in d[side]["sites"].items():
                if not s["ok"]:
                    site_class[sid] = -1
                    errors[sid] = s["error"]
                    continue
                for k in ("node_runs", "gate_calls", "verified"):
                    totals[side][k] += s[k]
                if s["mismatches"]:
                    # the node does not reduce to the lookup for this model sampling: never show an unverified row
                    site_class[sid] = -1
                    errors[sid] = "not shown: the page lookup could not be verified for this node on this model sampling"
                    excluded.append(f"{cid}/{side}/{sid}")
                    continue
                key = hashlib.sha1(json.dumps([s["k_start"], s["k_end"]]).encode()).hexdigest()
                if key not in index:
                    index[key] = len(classes)
                    classes.append(b64("H", s["k_start"] + s["k_end"]))
                site_class[sid] = index[key]
            sides[side] = {"bounds": b64("d", [float(x) for x in d[side]["bounds"]]), "classes": classes,
                           "site_class": site_class, "errors": errors}
        payload = {"id": cid, "family": p["family"], "shift": p["shift"], "ms_class": p["model_sampling_class"],
                   "n_sigmas": p["n_sigmas"], "sig": p["sigmas_f32"], "sched": sched,
                   "unavailable": p["schedulers_unavailable"], "sides": sides}
        js = f"PGE.data({json.dumps(cid)},{json.dumps(payload, separators=(',', ':'))});"
        with open(os.path.join(A.site, "data", f"{cid}.js"), "w") as f:
            f.write(js)
        payloads[cid] = js
        configs.append({"id": cid, "family": p["family"], "shift": p["shift"], "schedulers": list(sched), "n_sigmas": p["n_sigmas"]})
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problems; refusing to publish")

    sites = []
    for ms, ps in zip(meta["master"]["sites"], meta["pr"]["sites"]):
        assert ms["id"] == ps["id"]
        sites.append({"id": ps["id"], "label": ps["label"], "native": ps["native"], "sigma_source": ps["sigma_source"],
                      "requires_end_gt_start": ps["requires_end_gt_start"], "note": ps["note"],
                      "source": {"master": ms["source"], "pr": ps["source"]}})
    skip = ("RT_DETR", "SAM3", "DepthAnything", "Stable_Zero123")
    models = [m for m in meta["pr"]["models"] if not m["name"].startswith(skip)]
    out = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "repo": A.repo, "pr_number": A.pr_number, "tool_url": A.tool_url, "build_url": A.build_url,
        "commits": {side: meta[side]["commit"] for side in TREES},
        "torch": meta["pr"]["torch"], "python": meta["pr"]["python"], "max_steps": max_steps,
        "harness_sha256": hashlib.sha256(open(HARNESS, "rb").read()).hexdigest(),
        "schedulers": meta["pr"]["schedulers"], "family_label": meta["pr"]["family_label"],
        "models": models, "configs": configs, "sites": sites, "totals": totals, "excluded": excluded,
    }
    page = open(os.path.join(HERE, "page.html"), encoding="utf-8").read()
    html = page.replace("/*__META__*/null", json.dumps(out, separators=(",", ":")))
    with open(os.path.join(A.site, "index.html"), "w", encoding="utf-8") as f:
        f.write(html.replace("/*__INLINE_DATA__*/", ""))
    if A.single_file:
        with open(os.path.join(A.site, "single.html"), "w", encoding="utf-8") as f:
            f.write(html.replace("/*__INLINE_DATA__*/", "\n".join(payloads.values())))
    open(os.path.join(A.site, ".nojekyll"), "w").close()
    size = sum(os.path.getsize(os.path.join(A.site, "data", f)) for f in os.listdir(os.path.join(A.site, "data")))
    print(f"site written to {A.site}: {len(configs)} configs, {len(sites)} nodes, data {size / 1e6:.1f} MB")
    for side in TREES:
        print(f"  {side:6s} {out['commits'][side][:10]}  {totals[side]}")
    print(f"  excluded (lookup not verified): {excluded}")


if __name__ == "__main__":
    if not A.skip_measure:
        measure()
    merge()
