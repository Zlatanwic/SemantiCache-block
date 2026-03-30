"""Analyze Llama sanity check results by depth."""
import json
from collections import defaultdict

with open("results/ruler_niah/llama3b_full.json") as f:
    data = json.load(f)

for pol in ["full", "semantic", "snapkv"]:
    by_depth = defaultdict(list)
    for r in data:
        if r["policy"] == pol:
            by_depth[r["depth"]].append(100 if r["correct"] else 0)
    print(f"\n{pol}:")
    for d in sorted(by_depth):
        vals = by_depth[d]
        acc = sum(vals) / len(vals)
        n = sum(1 for v in vals if v)
        print(f"  d={d:.2f}: {acc:.0f}% ({n}/{len(vals)})")
