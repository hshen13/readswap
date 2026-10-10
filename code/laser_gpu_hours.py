#!/usr/bin/env python3
"""GPU hours of the LaSeR confirmatory jobs (time holding the GPU, not queueing).

Parses logs/*.log for the lines written by the exclusive GPU wrapper used in the original runs (GPU_RUN):
  [gpu_run <UTC time %Y-%m-%dT%H:%M:%SZ>] start: <command>
  [gpu_run <UTC time>] done rc=<exit code>: <command>
and keeps commands that contain the repository root. Writes results/analysis/gpu_hours.json.
The wrapper logs are not part of the release; results/analysis/gpu_hours.json holds the result.
"""
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

ROOT = os.environ.get("READSWAP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pat = re.compile(r"^\[gpu_run (\S+)\] (start|done rc=(\d+)): (.*)$")
open_jobs = {}
hours = defaultdict(float)
failed = defaultdict(float)
for f in sorted(glob.glob(f"{ROOT}/logs/*.log")):
    for line in open(f, errors="replace"):
        m = pat.match(line.strip())
        if not m or str(ROOT) not in m.group(4):
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
        cmd = m.group(4)
        if m.group(2) == "start":
            open_jobs[cmd] = t
            continue
        t0 = open_jobs.pop(cmd, None)
        if t0 is None:
            continue
        h = (t - t0).total_seconds() / 3600
        model = next((x for x in ("OctoThinker-3B-Short-LaSeR", "Qwen2.5-7B-LaSeR", "ORZ-7B-LaSeR") if x in cmd), "other")
        if "generate.py" in cmd:
            kind = "generation"
        elif "xchk" in cmd or "vllm_crosscheck" in cmd:
            kind = "crosscheck"
        elif "readout" in cmd:
            kind = "readout"
        else:
            kind = "other"
        if "/confirmatory/" not in cmd and kind != "crosscheck":
            kind = "dev_" + kind
        (hours if m.group(3) == "0" else failed)[f"{model}|{kind}"] += h
out = {"successful": {k: round(v, 2) for k, v in sorted(hours.items())},
       "failed_or_killed": {k: round(v, 2) for k, v in sorted(failed.items())}}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{ROOT}/results/analysis/gpu_hours.json", "w"), indent=1)
