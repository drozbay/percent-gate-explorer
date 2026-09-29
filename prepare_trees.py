"""Extract the two ComfyUI source trees the explorer compares.

Reads straight from git objects with `git archive`, so the working copy and its checked out branch
are never touched.

  python prepare_trees.py --repo /path/to/ComfyUI --pr-sha <sha> [--base-sha <sha>]

Without --base-sha the merge base of the PR commit and origin/master is used.
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="a ComfyUI git checkout that contains both commits")
ap.add_argument("--pr-sha", required=True)
ap.add_argument("--base-sha", default="")
ap.add_argument("--base-ref", default="origin/master")
ap.add_argument("--out", default=os.path.join(HERE, "trees"))
A = ap.parse_args()


def git(*args, binary=False):
    r = subprocess.run(["git", "-C", A.repo, *args], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr.decode(errors='replace')}")
    return r.stdout if binary else r.stdout.decode().strip()


pr = git("rev-parse", A.pr_sha + "^{commit}")
base = git("rev-parse", A.base_sha + "^{commit}") if A.base_sha else git("merge-base", pr, A.base_ref)

for side, sha in (("master", base), ("pr", pr)):
    dest = os.path.join(A.out, side)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", sha, binary=True))) as tar:
        tar.extractall(dest)
    with open(os.path.join(dest, "COMMIT"), "w") as f:
        f.write(sha + "\n")
    print(f"{side:6s} {sha}  {git('log', '-1', '--format=%cs %s', sha)[:80]}")

changed = git("diff", "--stat", base, pr).splitlines()
print(f"\n{changed[-1].strip() if changed else 'no differences'}")
