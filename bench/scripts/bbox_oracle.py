"""One-off pre-study analysis script (not part of the package); see bench/reports/prestudy-lean/review.md."""

# ruff: noqa: N806, N816, E741, E402, B905, B007, E501
import collections
import csv
import json
import math
import random
import re
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ann_dir, split_csv, out_json = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
objs = {}
for r in csv.DictReader(open(split_csv)):
    objs[r["object_key"]] = dict(type=r["fruit_type"], w=float(r["weight_g"]), split=r["split"])
pat = re.compile(r"^([a-z]+\d+)([ST])\((\d+)\)\.xml$")
imgs = collections.defaultdict(lambda: {"S": [], "T": []})
skipped = 0
for f in sorted(ann_dir.iterdir()):
    m = pat.match(f.name)
    if not m or m.group(1) not in objs:
        continue
    root = ET.parse(f).getroot()
    boxes = [
        (
            o.find("name").text,
            [int(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")],
        )
        for o in root.iter("object")
    ]
    coins = [b for n, b in boxes if n == "coin"]
    fruit = [b for n, b in boxes if n != "coin"]
    if len(coins) != 1 or len(fruit) != 1:
        skipped += 1
        continue
    c = coins[0]
    s = 25.0 / (((c[2] - c[0]) + (c[3] - c[1])) / 2)  # mm per px
    fw, fh = (fruit[0][2] - fruit[0][0]) * s, (fruit[0][3] - fruit[0][1]) * s
    imgs[m.group(1)][m.group(2)].append((fw, fh))


def proxy(key, mode):
    T, S = imgs[key]["T"], imgs[key]["S"]
    if mode == "top":
        if not T:
            return None
        return st.median([(a * b) ** 1.5 for a, b in T])
    if mode == "side":
        if not S:
            return None
        return st.median([a * b * min(a, b) for a, b in S])
    if not T or not S:
        return None
    a = st.median([max(x) for x in T])
    b = st.median([min(x) for x in T])
    h = st.median([y for _, y in S])
    return a * b * h


def evaluate(mode, single=False):
    keys = [k for k in objs if proxy(k, mode)]
    kt = collections.defaultdict(list)
    for k in keys:
        o = objs[k]
        if o["split"] == "dev":
            kt[o["type"]].append(math.log(o["w"] / proxy(k, mode)))
    kfit = {t: math.exp(st.mean(v)) for t, v in kt.items()}
    preds = [
        (objs[k]["type"], objs[k]["w"], kfit[objs[k]["type"]] * proxy(k, mode))
        for k in keys
        if objs[k]["split"] == "holdout" and objs[k]["type"] in kfit
    ]
    return preds


def metrics(preds):
    ape = [abs(p - t) / t for _, t, p in preds]
    lb = st.mean([math.log(p / t) for _, t, p in preds])
    # within-type beta
    by = collections.defaultdict(list)
    for ty, t, p in preds:
        by[ty].append((math.log(t), math.log(p)))
    sxx = sxy = 0
    for v in by.values():
        mx = st.mean(x for x, _ in v)
        my = st.mean(y for _, y in v)
        sxx += sum((x - mx) ** 2 for x, _ in v)
        sxy += sum((x - mx) * (y - my) for x, y in v)
    return dict(
        n=len(preds),
        mape=st.mean(ape),
        medape=st.median(ape),
        bias=math.exp(lb) - 1,
        beta=sxy / sxx if sxx else float("nan"),
    )


def boot(preds, f, n=2000, seed=7):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rnd.choice(preds) for _ in preds]
        try:
            vals.append(f(s))
        except ZeroDivisionError:
            pass
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


# B0
dev = collections.defaultdict(list)
for o in objs.values():
    if o["split"] == "dev":
        dev[o["type"]].append(o["w"])
b0 = [
    (o["type"], o["w"], st.mean(dev[o["type"]])) for o in objs.values() if o["split"] == "holdout"
]
out = {
    "images_used": {k: (len(v["T"]), len(v["S"])) for k, v in imgs.items()},
    "skipped_images": skipped,
}
rows = {}
for name, preds in [
    ("B0 type mean", b0),
    ("bbox top-only", evaluate("top")),
    ("bbox side-only", evaluate("side")),
    ("bbox top+side", evaluate("both")),
]:
    m = metrics(preds)
    m["mape_ci"] = boot(preds, lambda s: metrics(s)["mape"])
    m["beta_ci"] = boot(preds, lambda s: metrics(s)["beta"])
    rows[name] = m
    # excluding kiwi outlier object sensitivity
    m["mape_excl_min"] = metrics([x for x in preds if x[1] > 60])["mape"]
out["rows"] = rows
nT = sum(len(v["T"]) for v in imgs.values())
nS = sum(len(v["S"]) for v in imgs.values())
out["n_images"] = nT + nS
json.dump(out, open(out_json, "w"), indent=1)
for k, m in rows.items():
    print(
        f"{k:16s} n={m['n']:2d} MAPE={m['mape']:.3f} [{m['mape_ci'][0]:.3f},{m['mape_ci'][1]:.3f}] MedAPE={m['medape']:.3f} bias={m['bias']:+.3f} beta={m['beta']:.2f} [{m['beta_ci'][0]:.2f},{m['beta_ci'][1]:.2f}] MAPE w/o 52g kiwi={m['mape_excl_min']:.3f}"
    )
print("images", out["n_images"], "skipped", skipped)
# matched comparison on objects that have both views
keys = {(t, w) for t, w, _ in evaluate("both")}
b0m = [x for x in b0 if (x[0], x[1]) in keys]
bb = evaluate("both")
mb0, mbb = metrics(b0m), metrics(bb)
print(
    "matched n",
    len(b0m),
    "B0 MAPE",
    round(mb0["mape"], 3),
    "bbox MAPE",
    round(mbb["mape"], 3),
    "gain",
    round(1 - mbb["mape"] / mb0["mape"], 3),
)
# paired bootstrap of gain
pairs = list(zip(sorted(b0m, key=lambda x: (x[0], x[1])), sorted(bb, key=lambda x: (x[0], x[1]))))
rnd = random.Random(11)
g = []
for _ in range(2000):
    s = [rnd.choice(pairs) for _ in pairs]
    a = st.mean(abs(p - t) / t for _, t, p in (x[0] for x in s))
    b = st.mean(abs(p - t) / t for _, t, p in (x[1] for x in s))
    g.append(1 - b / a)
g.sort()
print("gain CI", round(g[50], 3), round(g[1950], 3))
per = collections.defaultdict(list)
for ty, t, p in bb:
    per[ty].append(abs(p - t) / t)
print({k: (len(v), round(st.mean(v), 3)) for k, v in per.items()})
print(
    "images per object (T,S) median",
    st.median([v["T"].__len__() for v in imgs.values()]),
    st.median([v["S"].__len__() for v in imgs.values()]),
)
