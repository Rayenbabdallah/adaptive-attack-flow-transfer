"""Reachability decomposition and paired cross-validation:
 (I) reachability decomposition: separate structurally-unreachable campaigns (no
     controls) from control-induced blocking; report static/adaptive success
     CONDITIONAL on no-controls reachability.
 (J) paired cross-validation: within-fold paired differences (exhaustive-greedy,
     chokepoint-greedy) over repeated fold partitions, at full and 3/4 coverage.
"""
from __future__ import annotations
import argparse, sys, importlib.util, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


HO = _load("held_out_v2", ROOT / "adaptive_transfer" / "held_out_v2.py")
X, E, R = HO.X, HO.E, HO.R
SEED = HO.SEED


def run(corpus, attack, out, k=3, folds=5, partitions=4):
    rng = np.random.default_rng(SEED)
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = HO.pool_from(camps)
    priors = rng.uniform(0.15, 0.85, size=(8, len(X.CLASSES)))
    comps = [X.gen_company(rng, priors, controls, s) for s in range(8) for _ in range(160 // 8)]
    covs = [X.covered(m["D"], controls) for m in comps]
    report = {}

    # (I) reachability decomposition
    reachable = [c for c in camps if X.succeeds(c, set(), pool, "static")]
    nR = len(reachable)
    def rate(camps_sub, mode):
        if not camps_sub: return float("nan")
        return float(np.mean([[X.succeeds(c, cov, pool, mode) for c in camps_sub] for cov in covs]))
    report["I_decomposition"] = {
        "n_campaigns": len(camps),
        "n_structurally_reachable_no_controls": nR,
        "frac_structurally_unreachable": round(1 - nR / len(camps), 3),
        "overall_static_success": round(rate(camps, "static"), 3),
        "overall_adaptive_success": round(rate(camps, "adaptive"), 3),
        "conditional_static_success_given_reachable": round(rate(reachable, "static"), 3),
        "conditional_adaptive_success_given_reachable": round(rate(reachable, "adaptive"), 3),
    }

    # (J) paired CV over repeated partitions
    for frac, tag in [(1.0, "full_coverage"), (0.75, "three_quarter_coverage")]:
        ctrl = HO.strict(controls, frac, np.random.default_rng(SEED + 1)) if frac < 1.0 else controls
        d_exh_g, d_chk_g = [], []
        for pz in range(partitions):
            pr = np.random.default_rng(1000 + pz)
            idx = pr.permutation(len(camps)); fi = np.array_split(idx, folds)
            for f in range(folds):
                ti = set(fi[f].tolist())
                train = [camps[i] for i in range(len(camps)) if i not in ti]
                test = [camps[i] for i in range(len(camps)) if i in ti]
                tp = HO.pool_from(train); pc = lambda c: HO.eval_pool(tp, c)
                bg = HO.breach(test, HO.greedy(train, ctrl, tp, k), ctrl, None, per_camp_pool=pc)
                bk = HO.breach(test, HO.choke(train, ctrl, tp, k), ctrl, None, per_camp_pool=pc)
                be = HO.breach(test, HO.exhaustive(train, ctrl, tp, k), ctrl, None, per_camp_pool=pc)
                d_exh_g.append(be - bg); d_chk_g.append(bk - bg)
        def summ(d):
            d = np.array(d)
            return {"mean_paired_diff": round(float(d.mean()), 3),
                    "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                    "frac_folds_negative": round(float(np.mean(d < 0)), 3), "n": len(d)}
        report[f"J_paired_{tag}"] = {"exhaustive_minus_greedy": summ(d_exh_g),
                                     "chokepoint_minus_greedy": summ(d_chk_g)}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "decomp.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(); run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
