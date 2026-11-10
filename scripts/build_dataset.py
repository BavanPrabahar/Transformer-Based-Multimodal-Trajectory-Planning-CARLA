"""Turn recorded expert episodes into (features, future trajectory) samples.

Filtering
---------
* samples whose 3 s future overlaps a noise-injection burst are dropped (the
  perturbed *history* is kept: those are the recovery examples),
* timed-out episodes are cut where the ego froze for good (> 6 s standstill to the end),
* collision episodes lose their final 4 s,
* the split into train/val is by episode (hash of the file name).

Usage: python scripts/build_dataset.py --data data/train --out data/features --stride 2
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmpc.geometry import Path, to_local  # noqa: E402
from tmpc.planning.features import FeatureConfig, build_features  # noqa: E402

CFG = FeatureConfig()


def valid_range(rec, status):
    v = rec["ego"][:, 3]
    n = len(v)
    end = n
    if status == "collision":
        end = max(0, n - 40)
    elif status in ("timeout", "offroad"):
        moving = np.where(v > 0.3)[0]
        last_move = int(moving[-1]) if len(moving) else 0
        if n - last_move > 60:          # frozen for > 6 s until the end
            end = min(n, last_move + 20)
        if status == "offroad":
            end = min(end, max(0, n - 20))
    return 0, end


def process(args):
    path, status, stride = args
    try:
        rec = dict(np.load(path, allow_pickle=False))
    except Exception as ex:
        return None, f"{path}: {ex}"
    route = Path(rec["route_xy"], rec["route_dl"], rec["route_dr"], rec["route_lw"])
    ego = rec["ego"]
    T = CFG.future_steps
    start, end = valid_range(rec, status)
    pert = rec["perturbed"].astype(bool)
    out = {k: [] for k in ["hist", "agents", "agents_mask", "route", "route_mask", "ctx", "target"]}
    v_des = float(rec["v_des"])
    for i in range(start, end - T, stride):
        if pert[i:i + T].any():
            continue
        fut = ego[i + 1:i + T + 1]
        if len(fut) < T:
            break
        f = build_features(ego[i, :3], ego[i, 3], ego[i, 4], ego[i, 5], ego[max(0, i - CFG.hist_len + 1):i + 1, :4],
                           rec["agents"][i], route, float(rec["s"][i]), v_des, CFG)
        tgt = to_local(fut[:, :2], ego[i, 0], ego[i, 1], ego[i, 2]).astype(np.float32)
        for k in f:
            out[k].append(f[k])
        out["target"].append(tgt)
    if not out["target"]:
        return None, None
    return {k: np.stack(v) for k, v in out.items()}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/train"])
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    jobs, meta = [], []
    for d in args.data:
        with open(os.path.join(d, "index.csv")) as f:
            for r in csv.DictReader(f):
                jobs.append((os.path.join(d, r["file"]), r["status"], args.stride))
                meta.append(r)
    print(f"{len(jobs)} episodes; statuses:", {s: sum(m['status'] == s for m in meta) for s in set(m['status'] for m in meta)})
    with Pool(args.workers) as pool:
        results = pool.map(process, jobs, chunksize=4)
    splits = {"train": [], "val": []}
    for (res, err), m in zip(results, meta):
        if err:
            print(err)
        if res is None:
            continue
        val = (zlib.crc32(m["file"].encode()) % 1000) / 1000.0 < args.val_frac
        res["template"] = np.array([m["template"]] * len(res["target"]))
        res["episode"] = np.array([m["file"]] * len(res["target"]))
        splits["val" if val else "train"].append(res)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for name, parts in splits.items():
        if not parts:
            continue
        arrs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(f"{args.out}_{name}.npz", **arrs)
        print(name, "samples:", len(arrs["target"]), "episodes:", len(parts))
    json.dump(CFG.to_dict(), open(f"{args.out}_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
