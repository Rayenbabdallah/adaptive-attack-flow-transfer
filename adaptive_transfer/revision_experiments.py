"""Substitution-feasibility ablation, campaign bootstrap, and exhaustive optimizer:
 (A) realism-constrained substitution ablation (platform-matched, co-occurrence-matched)
     -- does the static-vs-adaptive gap survive feasibility restrictions?
 (B) campaign-level bootstrap CI (resample the 34 campaigns) -- corpus uncertainty.
 (C) exhaustive optimal k=3 portfolio vs greedy -- is greedy's failure the optimizer's
     fault (a better optimizer closes it) or structural?
"""
from __future__ import annotations
import argparse, json, sys, importlib.util, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realism"))


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


E = _load("experiment", ROOT / "defense_surface" / "experiment.py")
X = _load("xfer_experiment", ROOT / "adaptive_transfer" / "transfer_engine.py")
R = X.R
SEED = X.SEED


def load_platforms(attack):
    objs = json.load(open(attack, encoding="utf-8"))["objects"]
    plat = {}
    for o in objs:
        if o.get("type") != "attack-pattern" or o.get("revoked"):
            continue
        ext = next((r["external_id"] for r in o.get("external_references", [])
                    if r.get("source_name") == "mitre-attack" and "external_id" in r), None)
        if ext and "." not in ext:
            plat[ext] = set(o.get("x_mitre_platforms", []) or [])
    return plat


def succeeds_constrained(camp, cov, allowed_fn, mode):
    tac_tech = defaultdict(list)
    for t, ta in zip(camp.techniques, camp.tactics):
        tac_tech[ta].append(t)
    ach = set()
    for ta in set(camp.tactics):
        orig = tac_tech[ta]
        if mode == "static":
            if any(t not in cov for t in orig):
                ach.add(ta)
        else:
            subs = allowed_fn(ta, orig)
            if any(t not in cov for t in subs):
                ach.add(ta)
    obj = X.objective_of(camp)
    if obj is None or obj not in ach:
        return False
    return X.reachable(ach, obj)


def build(corpus, attack):
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls = E.load_attack(attack)
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    pool = {k: sorted(v) for k, v in pool.items()}
    plat = load_platforms(attack)
    # co-occurrence: techniques appearing together in some campaign
    cooc = defaultdict(set)
    for c in camps:
        ts = set(c.techniques)
        for t in ts:
            cooc[t] |= ts
    return camps, controls, pool, plat, cooc


def allowed_fns(pool, plat, cooc):
    def corpus(ta, orig): return pool.get(ta, [])
    def platform(ta, orig):
        ops = set().union(*[plat.get(o, set()) for o in orig]) if orig else set()
        if not ops:  # unknown platform -> fall back to corpus pool (conservative)
            return pool.get(ta, [])
        return [t for t in pool.get(ta, []) if (plat.get(t, set()) & ops) or not plat.get(t, set())]
    def cooccur(ta, orig):
        allow = set().union(*[cooc.get(o, set()) for o in orig]) if orig else set()
        return [t for t in pool.get(ta, []) if t in allow]
    return {"corpus": corpus, "platform": platform, "cooccur": cooccur}


def run(corpus, attack, out, n_comp=160):
    rng = np.random.default_rng(SEED)
    camps, controls, pool, plat, cooc = build(corpus, attack)
    fns = allowed_fns(pool, plat, cooc)
    priors = rng.uniform(0.15, 0.85, size=(8, len(X.CLASSES)))
    comps = [X.gen_company(rng, priors, controls, s) for s in range(8) for _ in range(n_comp // 8)]
    covs = [X.covered(m["D"], controls) for m in comps]
    report = {"n_companies": len(comps), "n_campaigns": len(camps)}

    # (A) realism-constrained substitution ablation
    abl = {}
    stat = np.mean([[succeeds_constrained(c, cov, None, "static") for c in camps] for cov in covs])
    for name, fn in fns.items():
        ad = np.mean([[succeeds_constrained(c, cov, fn, "adaptive") for c in camps] for cov in covs])
        abl[name] = {"adaptive_success": float(ad), "gap_vs_static": float(ad - stat)}
    abl["static_success"] = float(stat)
    report["A_substitution_ablation"] = abl

    # (B) campaign-level bootstrap of the gap (corpus pool), resampling campaigns
    S = np.array([[succeeds_constrained(c, cov, None, "static") for c in camps] for cov in covs])      # orgs x camps
    A = np.array([[succeeds_constrained(c, cov, fns["corpus"], "adaptive") for c in camps] for cov in covs])
    gaps = []
    for _ in range(2000):
        idx = rng.integers(0, len(camps), len(camps))
        gaps.append(float(A[:, idx].mean() - S[:, idx].mean()))
    report["B_campaign_bootstrap_gap"] = {"mean": float((A.mean() - S.mean())),
                                          "ci_campaign_resample": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]}

    # (C) exhaustive optimal k=3 vs greedy, population-level, evaluated on adaptive (corpus) truth
    appl = [m for m, c in controls.items() if True]  # all controls (population view)
    fn = fns["corpus"]

    def adaptive_breach(D):
        cov = X.covered(D, controls)
        return float(np.mean([succeeds_constrained(c, cov, fn, "adaptive") for c in camps]))

    def static_breach(D):
        cov = X.covered(D, controls)
        return float(np.mean([succeeds_constrained(c, cov, None, "static") for c in camps]))
    # greedy on adaptive objective
    def greedy(obj, k):
        chosen = []
        for _ in range(k):
            best, bm = None, None
            for mid in appl:
                if mid in chosen: continue
                v = obj(chosen + [mid])
                if best is None or v < best: best, bm = v, mid
            chosen.append(bm)
        return chosen
    k = 3
    g_adapt = greedy(adaptive_breach, k)
    g_static = greedy(static_breach, k)
    # exhaustive optimal for adaptive breach
    best_val, best_set = 1e9, None
    for combo in itertools.combinations(appl, k):
        v = adaptive_breach(list(combo))
        if v < best_val: best_val, best_set = v, combo
    report["C_optimizer_k3"] = {
        "greedy_adaptive_breach": adaptive_breach(g_adapt),
        "greedy_static_rule_adaptive_breach": adaptive_breach(g_static),
        "exhaustive_optimal_adaptive_breach": best_val,
        "exhaustive_beats_greedy_by": adaptive_breach(g_adapt) - best_val,
        "no_controls_adaptive_breach": adaptive_breach([]),
        "n_subsets_evaluated": sum(1 for _ in itertools.combinations(appl, k)),
    }

    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "revision.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(); run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
