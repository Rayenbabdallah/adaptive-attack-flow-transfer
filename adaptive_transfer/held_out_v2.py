"""Leakage-free held-out validation + quantitative audits:
 (F) 5-fold CV where the adaptive SUBSTITUTE POOL is rebuilt from TRAINING campaigns
     only (no held-out leakage); attacker on a held-out campaign may use the
     train-derived pool plus that campaign's own recorded techniques. Report per-fold
     selected controls and held-out breach, at full and reduced coverage.
 (G) cut prevalence: fraction of campaigns with >=1 required-tactic cut, and fraction
     whose cut pool is fully coverable by some control.
 (H) quantitative substitution audit: random sample of co-occurrence substitutions,
     fraction platform-compatible (a necessary-condition proxy for interchangeability).
"""
from __future__ import annotations
import argparse, sys, importlib.util, itertools, json
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


def pool_from(camps):
    pool = defaultdict(set)
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            pool[ta].add(t)
    return {k: sorted(v) for k, v in pool.items()}


def eval_pool(train_pool, camp):
    """attacker pool on a held-out campaign: train-derived pool UNION this campaign's own
    recorded techniques per tactic (the attacker always has what the report records)."""
    p = {k: set(v) for k, v in train_pool.items()}
    for t, ta in zip(camp.techniques, camp.tactics):
        p.setdefault(ta, set()).add(t)
    return {k: sorted(v) for k, v in p.items()}


def succ_adaptive(camp, cov, pool):
    return X.succeeds(camp, cov, pool, "adaptive")


def strict(controls, frac, rng):
    out = {}
    for mid, c in controls.items():
        cov = sorted(c["cover"])
        keep = set(cov) if frac >= 1.0 else (set(rng.choice(cov, size=max(1, int(len(cov) * frac)), replace=False)) if cov else set())
        out[mid] = {"name": c["name"], "cover": set(keep), "class": c["class"]}
    return out


def required_tactics(camp):
    obj = X.objective_of(camp)
    if obj is None:
        return []
    tacs = {t for t in camp.tactics if t in X.STATE_MODEL}
    return [t for t in tacs if not X.reachable(tacs - {t}, obj)]


def breach(camps, D, controls, pool_sel, per_camp_pool=None):
    cov = X.covered(D, controls)
    vals = []
    for c in camps:
        pl = per_camp_pool(c) if per_camp_pool else pool_sel
        vals.append(succ_adaptive(c, cov, pl))
    return float(np.mean(vals))


def greedy(train, controls, pool, k):
    appl = list(controls); chosen = []
    for _ in range(k):
        bm, bv = None, None
        for m in appl:
            if m in chosen: continue
            v = breach(train, chosen + [m], controls, pool)
            if bv is None or v < bv: bv, bm = v, m
        chosen.append(bm)
    return chosen


def choke(train, controls, pool, k):
    cps = []
    for c in train:
        cps.append([(t, pool.get(t, [])) for t in required_tactics(c) if pool.get(t)])
    appl = list(controls); chosen = []
    def sur(D):
        cov = X.covered(D, controls); tot = 0.0
        for cp in cps:
            best = 0.0
            for ta, pl in cp:
                if pl: best = max(best, sum(x in cov for x in pl) / len(pl))
            tot += best
        return tot
    for _ in range(k):
        bm, bv = None, None
        for m in appl:
            if m in chosen: continue
            v = sur(chosen + [m])
            if bv is None or v > bv: bv, bm = v, m
        chosen.append(bm)
    return chosen


def exhaustive(train, controls, pool, k):
    best, bs = None, None
    for combo in itertools.combinations(list(controls), k):
        v = breach(train, list(combo), controls, pool)
        if best is None or v < best: best, bs = v, list(combo)
    return bs


def run(corpus, attack, out, k=3, folds=5):
    rng = np.random.default_rng(SEED)
    camps = R.load_corpus(corpus, min_len=4)
    ta_techs, controls_full = E.load_attack(attack)
    plat = load_platforms(attack)
    cooc = defaultdict(set)
    for c in camps:
        ts = set(c.techniques)
        for t in ts: cooc[t] |= ts
    idx = rng.permutation(len(camps)); folds_idx = np.array_split(idx, folds)
    report = {"k": k, "folds": folds, "n_campaigns": len(camps)}

    # comparability: no-controls adaptive success under the FULL-CORPUS pool (vs train-derived)
    pool_all0 = pool_from(camps)
    report["no_controls_full_corpus_pool"] = float(np.mean([succ_adaptive(c, set(), pool_all0) for c in camps]))

    for frac, tag in [(1.0, "full_coverage"), (0.75, "three_quarter_coverage"), (0.5, "half_coverage")]:
        controls = strict(controls_full, frac, np.random.default_rng(SEED + 1)) if frac < 1.0 else controls_full
        res = {m: [] for m in ("no_controls", "greedy", "chokepoint", "exhaustive")}
        picks = {"greedy": [], "chokepoint": [], "exhaustive": []}
        for f in range(folds):
            ti = set(folds_idx[f].tolist())
            train = [camps[i] for i in range(len(camps)) if i not in ti]
            test = [camps[i] for i in range(len(camps)) if i in ti]
            tp = pool_from(train)                      # TRAIN-ONLY pool (no leakage)
            pc = lambda c: eval_pool(tp, c)            # held-out eval pool
            pg = greedy(train, controls, tp, k)
            pk = choke(train, controls, tp, k)
            pe = exhaustive(train, controls, tp, k)
            res["no_controls"].append(breach(test, [], controls, None, per_camp_pool=pc))
            res["greedy"].append(breach(test, pg, controls, None, per_camp_pool=pc))
            res["chokepoint"].append(breach(test, pk, controls, None, per_camp_pool=pc))
            res["exhaustive"].append(breach(test, pe, controls, None, per_camp_pool=pc))
            picks["greedy"].append([controls[m]["name"] for m in pg])
            picks["chokepoint"].append([controls[m]["name"] for m in pk])
            picks["exhaustive"].append([controls[m]["name"] for m in pe])
        report[tag] = {"mean": {m: float(np.mean(v)) for m, v in res.items()},
                       "sd_across_folds": {m: float(np.std(v)) for m, v in res.items()},
                       "per_fold": {m: [round(x, 3) for x in v] for m, v in res.items()},
                       "selected_controls_per_fold": picks}

    # (G) cut prevalence (full coverage, full-corpus pool for structural property)
    pool_all = pool_from(camps)
    all_cov_union = set().union(*[controls_full[m]["cover"] for m in controls_full])
    has_cut = 0; coverable = 0
    for c in camps:
        req = required_tactics(c)
        if req: has_cut += 1
        if any(pool_all.get(t) and set(pool_all[t]).issubset(all_cov_union) for t in req):
            coverable += 1
    report["cut_prevalence"] = {"campaigns_with_required_tactic_cut": has_cut,
                                "campaigns_with_coverable_cut": coverable, "n": len(camps)}

    # (H) quantitative substitution audit: random co-occurrence substitutions, platform-compatible?
    ra = np.random.default_rng(7)
    pairs = []
    for c in camps:
        for t, ta in zip(c.techniques, c.tactics):
            for s in pool_all.get(ta, []):
                if s != t and s in cooc[t]:
                    pairs.append((t, s))
    uniq = sorted(set(pairs))  # sorted: set order varies with PYTHONHASHSEED
    samp = [uniq[i] for i in ra.choice(len(uniq), size=min(40, len(uniq)), replace=False)]
    platcompat = sum(1 for (t, s) in samp if (plat.get(t, set()) & plat.get(s, set())) or not plat.get(t) or not plat.get(s))
    report["substitution_audit_quant"] = {"n_unique_cooccurrence_substitutions": len(uniq),
                                           "n_sampled": len(samp),
                                           "criterion": "shares >=1 ATT&CK platform (necessary-condition proxy)",
                                           "n_platform_compatible": int(platcompat),
                                           "frac_platform_compatible": round(platcompat / len(samp), 3)}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "held_out_v2.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--attack", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args(); run(a.corpus, a.attack, a.out)


if __name__ == "__main__":
    main()
